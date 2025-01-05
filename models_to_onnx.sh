paddle2onnx --model_dir ./models/en_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./models/det_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./models/en_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./models/rec_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./models/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./models/cls_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True