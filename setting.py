# ------------- teacher net --------------------#
# Optional pretrained checkpoint paths for legacy 2D classification teachers.
# These entries are not required for the 3D segmentation pipeline used in
# `train_student_rl_new_policy.py` (the 3D pipeline loads teacher *features*
# directly from disk, configured by the `teachers_features_paths` block in
# the dataset YAML).
#
# If you want to use one of the 2D teacher backbones, fill in your local
# checkpoint paths below. Otherwise this dictionary can stay empty.
teacher_model_path_dict = {
    # 'RegNetY_400MF':   '<PATH_TO>/RegNetY_400MF_best.pth',
    # 'RegNetX_400MF':   '<PATH_TO>/RegNetX_400MF_best.pth',
    # 'resnet32x4':      '<PATH_TO>/resnet32x4_best.pth',
    # 'resnet110x2':     '<PATH_TO>/resnet110x2_best.pth',
    # 'wrn_28_4':        '<PATH_TO>/wrn_28_4_best.pth',
    # 'ResNet50':        '<PATH_TO>/resnet50.pth',
    # 'ResNet101':       '<PATH_TO>/resnet101.pth',
    # 'wide_resnet50_2': '<PATH_TO>/wide_resnet50_2.pth',
    # 'resnext50_32x4d': '<PATH_TO>/resnext50_32x4d.pth',
}
