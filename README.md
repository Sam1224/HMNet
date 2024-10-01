# Hybrid Mamba for Few-Shot Segmentation

This repository contains the code for our NIPS 2024 [paper](https://arxiv.org/abs/2409.19613) "*Hybrid Mamba for Few-Shot Segmentation*", where we design a cross attention-like Mamba method to enable support-query interactions.

> **Abstract**: *Many few-shot segmentation (FSS) methods use cross attention to fuse support foreground (FG) into query features, regardless of the quadratic complexity. A recent advance Mamba can also well capture intra-sequence dependencies, yet the complexity is only linear. Hence, we aim to devise a cross (attention-like) Mamba to capture inter-sequence dependencies for FSS. A simple idea is to scan on support features to selectively compress them into the hidden state, which is then used as the initial hidden state to sequentially scan query features. Nevertheless, it suffers from (1) support forgetting issue: query features will also gradually be compressed when scanning on them, so the support features in hidden state keep reducing, and many query pixels cannot fuse sufficient support features; (2) intra-class gap issue: query FG is essentially more similar to itself rather than support FG, i.e., query may prefer not to fuse support but their own features from the hidden state, yet the effective use of support information leads to the success of FSS. To tackle them, we design a hybrid Mamba network (HMNet), including (1) a support recapped Mamba to periodically recap the support features when scanning query, so the hidden state can always contain rich support information; (2) a query intercepted Mamba to forbid the mutual interactions among query pixels, and encourage them to fuse more support features from the hidden state. Consequently, the support information is better utilized, leading to better performance. Extensive experiments have been conducted on two public benchmarks, showing the superiority of HMNet.*

## Dependencies

- Python 3.10
- PyTorch 1.12.0
- cuda 11.6
- torchvision 0.13.0
```
> conda env create -f env.yaml
```

## Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

You can download the pre-processed PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup> datasets [here](https://entuedu-my.sharepoint.com/:f:/g/personal/qianxion001_e_ntu_edu_sg/ErEg1GJF6ldCt1vh00MLYYwBapLiCIbd-VgbPAgCjBb_TQ?e=ibJ4DM), and extract them into `data/` folder. Then, you need to create a symbolic link to the `pascal/VOCdevkit` data folder as follows:
```
> ln -s <absolute_path>/data/pascal/VOCdevkit <absolute_path>/data/VOCdevkit2012
```

The directory structure is:

    ../
    ├── HMNet/
    └── data/
        ├── VOCdevkit2012/
        │   └── VOC2012/
        │       ├── JPEGImages/
        │       ├── ...
        │       └── SegmentationClassAug/
        └── MSCOCO2014/           
            ├── annotations/
            │   ├── train2014/ 
            │   └── val2014/
            ├── train2014/
            └── val2014/

## Models

- Download the pretrained backbones from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EUHlKdET3mJGie_IjtpzW5kBo45yz0PB2dW9n55Vo5acXw?e=uyuUDX) and put them into the `initmodel` directory.
- Download [exp.tar.gz](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EbcNC1Ram0lJozZ2qe624uEBhXNmKrI64CM0uhEPJuxaig?e=iDMvrI) to obtain all trained models for PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>.

## Testing

- **Commands**:
  ```
  sh test_pascal.sh {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}
  sh test_coco.sh {Split: 0/1/2/3} {Net: resnet50/vgg} {Postfix: manet/manet_5s}

  # e.g., testing split 0 under 1-shot setting on PASCAL-5<sup>i</sup>, with ResNet50 as the pretrained backbone:
  sh test_pascal.sh 0 resnet50 manet
  
  # e.g., testing split 0 under 5-shot setting on COCO-20<sup>i</sup>, with ResNet50 as the pretrained backbone:
  sh test_coco.sh 0 resnet50 manet_5s
  ```

## References

This repo is mainly built based on [BAM](https://github.com/chunbolang/BAM). Thanks for their great work!

