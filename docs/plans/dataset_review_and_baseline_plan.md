# Dataset Review and Baseline Training Plan

## Goal

Prepare a reproducible visual review package for the fixed four-class dataset,
then implement and smoke-test a shared ResNet50/EfficientNet-B0 training
pipeline without changing the existing train/val/test assignments.

## Tasks

1. Create a deterministic mineral-stratified review sample:
   - Up to 50 images per mineral.
   - Include train, validation, and test records when available.
   - Include all eligible titanomagnetite images.
   - Generate a review CSV and contact sheets.
2. Verify the local Python, PyTorch, torchvision, CUDA, and GPU environment.
3. Test manifest loading, path resolution, class mapping, and fixed split use.
4. Implement one shared training entry point for ResNet50 and EfficientNet-B0.
5. Run automated tests and a bounded smoke training run.
6. Preserve logs, metrics, configuration, and review instructions.

## Constraints

- Do not reshuffle or regenerate `dataset_split_manifest.csv`.
- Do not delete or move original images.
- Use `Macro-F1` and target-class recall as primary evaluation metrics.
- Treat Mindat specimen images as proxy mineral images, not industrial conveyor
  ore images.
