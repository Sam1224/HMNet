import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model.ASPP import ASPP
from model.backbone_utils import Backbone
from model.mamba_blocks import VSSM
from model.loss import WeightedDiceLoss
from model.PSPNet import OneModel as PSPNet
from einops import rearrange


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, blocks=None):
        super(OneModel, self).__init__()
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion_dice = WeightedDiceLoss()
        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 2
        
        # Feature size
        self.size = (80, 80) if args.train_h > 473 else (64, 64)

        # BAM's setting
        self.cls_type = "Base"  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
            
        assert self.layers in [50, 101, 152]
        PSPNet_ = PSPNet(args)
        pre_weight = "./initmodel/PSPNet/{}/split{}/{}/best.pth".format(self.dataset, args.split, "resnet{}".format(args.layers) if not args.vgg else "vgg")
        print(pre_weight)
        new_param = torch.load(pre_weight, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                 
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learner =  nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        # Meta Learner
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        embed_dim = reduce_dim
        # embed_dim = 64  # amnet
        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 2, embed_dim, kernel_size=1, padding=0, bias=False),
        )
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, embed_dim, kernel_size=1, padding=0, bias=False),
        )

        # VMamba
        depths = [blocks] if blocks else [8]
        dims = [embed_dim]
        mlp_ratio = 1 if embed_dim == reduce_dim else 4
        self.mamba = VSSM(
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio
        )

        scale = 0
        for i in range(len(depths)):
            scale += 2 ** i
        self.ASPP_meta = ASPP(scale * embed_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(scale * embed_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Gram Merge        
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))
        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))
        
        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

    def get_optim(self, model, args, LR):
        base_bs = 2
        ratio = args.batch_size // base_bs

        params = [
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge_query.parameters()},
                {'params': model.init_merge_supp.parameters()},
                {'params': model.ASPP_meta.parameters()},
                {'params': model.res1_meta.parameters()},
                {'params': model.res2_meta.parameters()},
                {'params': model.cls_meta.parameters()},
                {'params': model.gram_merge.parameters()},
                {'params': model.cls_merge.parameters()},
            ]
        if self.shot > 1:
            params.append({'params': model.kshot_rw.parameters()})
        optimizer = torch.optim.SGD(params, lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_swin = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "mamba" in n and "norm" not in n and p.requires_grad]},
                {"params": [p for n, p in model.named_parameters() if "mamba" in n and "norm" in n and p.requires_grad], "weight_decay": 0.}
            ], lr=6e-5 * ratio
        )
        return optimizer, optimizer_swin

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learner.parameters():
            param.requires_grad = False

    def generate_prior(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        fg_sim_maxs = []
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat = tmp_supp_feat
            q = query_feat_high.flatten(2).transpose(-2, -1)
            s = tmp_supp_feat.flatten(2).transpose(-2, -1)

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            fg_sim_max = similarity.max(1)[0]  # bsize
            fg_sim_maxs.append(fg_sim_max.unsqueeze(-1))  # bsize, 1
            
            # similarity = similarity.permute(0, 2, 1)
            # similarity = F.softmax(similarity, dim=-1)
            # similarity = torch.bmm(similarity, tmp_mask.flatten(2).transpose(-2, -1)).squeeze(-1)
            # similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
            #         similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            # corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)   
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            
            corr_query = F.interpolate(corr_query, size=fts_size, mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)  # bsize, shots
        return corr_query_mask, fg_sim_maxs

    def extract_feats(self, x):
        results = {}
        with torch.no_grad():
            feat = self.layer0(x)
            feat = self.layer1(feat)
            layers = [self.layer2, self.layer3, self.layer4]
            for idx, layer in enumerate(layers):
                feat = layer(feat)
                results[str(idx + 1)] = feat.clone()
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results["4"] = (feat)
        return results

    def cos_sim(self, query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        return similarity

    def generate_prior_proto(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        fg_list = []
        bg_list = []
        fg_sim_maxs = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_supp_feat = Weighted_GAP(tmp_supp_feat, 1 - tmp_mask)

            fg_sim = self.cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
            bg_sim = self.cos_sim(query_feat_high, bg_supp_feat, cosine_eps)

            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            
            fg_sim_max = fg_sim.max(1)[0]  # bsize
            fg_sim_maxs.append(fg_sim_max.unsqueeze(-1))  # bsize, 1

            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                        fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                    bg_sim.max(1)[0].unsqueeze(1) - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)

            fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
            bg_sim = F.interpolate(bg_sim, size=fts_size, mode='bilinear', align_corners=True)
            fg_list.append(fg_sim)
            bg_list.append(bg_sim)
        fg_corr = torch.cat(fg_list, 1)  # bsize, shots, h, w
        bg_corr = torch.cat(bg_list, 1)
        corr = (fg_corr - bg_corr)
        corr[corr < 0] = 0
        corr_max = corr.view(bsize, len(final_supp_list), -1).max(-1)[0]  # bsize, shots
        
        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)  # bsize, shots
        return fg_corr, bg_corr, corr, fg_sim_maxs, corr_max

    # que_img, sup_img, sup_mask, que_mask(meta), cat_idx(meta)
    @autocast()
    def forward(self, x, s_x, s_y, y_m, cat_idx=None):
        x_size = x.size()  # bs, 3, 473, 473
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)  # (473 - 1) / 8 * 8 + 1 = 60
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)  # 60

        # Interpolation size for pascal/coco
        size = self.size

        # ========================================
        # Feature Extraction - Query/Support
        # ========================================
        # Query/Support Feature
        with torch.no_grad():
            qry_bcb_fts = self.extract_feats(x)
            supp_bcb_fts = self.extract_feats(s_x.view(-1, 3, x_size[2], x_size[3]))

        if self.vgg:
            qry_bcb_fts['1'] = F.interpolate(qry_bcb_fts['1'], size=qry_bcb_fts['2'].size()[-2:], mode='bilinear', align_corners=True)
            supp_bcb_fts['1'] = F.interpolate(supp_bcb_fts['1'], size=supp_bcb_fts['2'].size()[-2:], mode='bilinear', align_corners=True)

        query_feat_high_4 = qry_bcb_fts['3']
        query_feat_high_5 = qry_bcb_fts['4']

        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        query_feat = self.down_query(query_feat)
        query_feat = F.interpolate(query_feat, size=size, mode='bilinear', align_corners=True)
        fts_size = query_feat.size()[-2:]
        supp_feat = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat = F.interpolate(supp_feat, size=size, mode='bilinear', align_corners=True)

        mask_list = []
        supp_pro_list = []
        supp_feat_list = []
        final_supp_list_4 = []
        final_supp_list_5 = []
        supp_feat_mid = supp_feat.view(bs, self.shot, -1, fts_size[0], fts_size[1])
        supp_bcb_fts['3'] = F.interpolate(supp_bcb_fts['3'], size=size, mode='bilinear', align_corners=True)
        supp_feat_high_4 = supp_bcb_fts['3'].view(bs, self.shot, -1, fts_size[0], fts_size[1])
        supp_bcb_fts['4'] = F.interpolate(supp_bcb_fts['4'], size=size, mode='bilinear', align_corners=True)
        supp_feat_high_5 = supp_bcb_fts['4'].view(bs, self.shot, -1, fts_size[0], fts_size[1])
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=fts_size, mode='bilinear', align_corners=True)
            mask_list.append(mask)
            final_supp_list_4.append(supp_feat_high_4[:, i, :, :, :])
            final_supp_list_5.append(supp_feat_high_5[:, i, :, :, :])
            supp_feat_list.append((supp_feat_mid[:, i, :, :, :] * mask).unsqueeze(-1))
            supp_pro = Weighted_GAP(supp_feat_mid[:, i, :, :, :], mask)
            supp_pro_list.append(supp_pro)

        # Support features/prototypes/masks
        supp_mask = torch.cat(mask_list, dim=1).mean(1, True)  # bs, 1, 60, 60
        supp_feat = torch.cat(supp_feat_list, dim=-1).mean(-1)  # bs, 256, 60, 60
        supp_pro = torch.cat(supp_pro_list, dim=2).mean(2, True)  # bs, 256, 1, 1
        supp_pro = supp_pro.expand_as(query_feat)  # bs, 256, 60, 60

        # Prior Similarity Mask
        corr_fg_4, _, corr_4, corr_fg_4_sim_max, corr_4_sim_max = self.generate_prior_proto(query_feat_high_4, final_supp_list_4, mask_list, fts_size)
        corr_fg_5, _, corr_5, corr_fg_5_sim_max, corr_5_sim_max = self.generate_prior_proto(query_feat_high_5, final_supp_list_5, mask_list, fts_size)
                
        corr_fg = corr_fg_4.clone()  # bs, shots, h, w
        corr = corr_4.clone()  # bs, shots, h, w
        for i in range(bs):
            for j in range(self.shot):
                if corr_fg_4_sim_max[i, j] < corr_fg_5_sim_max[i, j]:
                    corr_fg[i, j] = corr_fg_5[i, j]
                if corr_4_sim_max[i, j] < corr_5_sim_max[i, j]:
                    corr[i, j] = corr_5[i, j]
        corr_fg = corr_fg.mean(1, True)
        corr = corr.mean(1, True)
        corr_query_mask = torch.cat([corr_fg, corr], dim=1)
        
        # ========================================
        # Mamba
        # ========================================
        # Adapt query/support features with support prototype
        query_cat = torch.cat([query_feat, supp_pro, corr_query_mask * 10], dim=1)  # bs, 512, 60, 60
        query_feat = self.init_merge_query(query_cat)  # bs, 256, 60, 60
        supp_cat = torch.cat([supp_feat, supp_pro, supp_mask], dim=1)  # bs, 512, 60, 60
        supp_feat = self.init_merge_supp(supp_cat)  # bs, 256, 60, 60

        # mamba blocks
        query_feat = self.mamba(query_feat, supp_feat)
        merge_feat = self.relu(query_feat)

        # ========================================
        # Meta Output
        # ========================================
        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)
        
        # ========================================
        # Base Output
        # ========================================
        base_out = F.interpolate(qry_bcb_fts['4'], size=size, mode='bilinear', align_corners=True)
        base_out = self.base_learner(base_out)
        
        # ========================================
        # Output Ensemble
        # ========================================
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(qry_bcb_fts['1']) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        supp_feat_list = rearrange(supp_bcb_fts['1'], "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list = [supp_feat_list[:, i, ...] for i in range(self.shot)]
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True) # [bs, 1, 1, 1]      

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM ) 
        meta_map_bg = meta_out_soft[:, 0:1, :, :]                           
        meta_map_fg = meta_out_soft[:, 1:, :, :]                            
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list,0)
        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], dim=1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Interpolate
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # ========================================
        # Loss
        # ========================================
        if self.training:
            main_loss = self.criterion_ce(final_out, y_m.long())
            aux_loss1 = self.criterion_dice(meta_out, y_m.long())
            aux_loss2 = torch.zeros_like(main_loss)
            
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out
