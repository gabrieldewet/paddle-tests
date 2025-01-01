python3 tools/infer/predict_system.py --use_gpu=False \
--cls_model_dir=./inference/ch_ppocr_mobile_v2.0_cls_infer \
--rec_model_dir=./inference/en_PP-OCRv3_rec_infer \
--det_model_dir=./inference/en_PP-OCRv3_det_infer \
--image_dir=./docs/ppocr/infer_deploy/images/img_12.jpg\
--rec_char_dict_path=ppocr/utils/en_dict.txt