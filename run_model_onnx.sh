uv run tools/infer/predict_system.py --use_gpu=True 
--use_onnx=True \
--det_model_dir=./models/det_onnx/model.onnx  \
--rec_model_dir=./models/rec_onnx/model.onnx  \
--cls_model_dir=./models/cls_onnx/model.onnx  \
--image_dir=./data/release_document_signed.pdf