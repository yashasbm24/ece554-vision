# COCO 2017 validation subset

105 unique validation photographs and YOLO-format bounding boxes. Validation only.

Selected 15 images per target class from the first mirrored validation shard using seed 554, then deduplicated. All original detection labels for selected images were retained. This targeted subset does not represent the full COCO validation distribution.

- person: 64 images
- bicycle: 15 images
- car: 32 images
- traffic light: 16 images
- stop sign: 15 images
- bench: 17 images
- dog: 16 images
- backpack: 17 images
- chair: 16 images

Images: https://huggingface.co/datasets/ariG23498/coco2017
Labels: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
Dataset and individual photo license information: https://cocodataset.org/#termsofuse
