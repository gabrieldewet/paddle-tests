DET_MODEL=Multilingual_PP-OCRv3_det_infer
CLS_MODEL=ch_ppocr_mobile_v2.0_cls_infer
REC_MODEL=latin_PP-OCRv3_rec_infer
REC_TXT=latin_dict.txt

# echo "Converting $DET_MODEL to ONNX (paddle) format"
# tar -xf ./models/paddle/$DET_MODEL.tar -C ./models/paddle/
# paddle2onnx --model_dir ./models/paddle/$DET_MODEL \
# --model_filename inference.pdmodel \
# --params_filename inference.pdiparams \
# --save_file ./models/paddle_onnx/$DET_MODEL.onnx \
# --opset_version 11 \
# --enable_onnx_checker True

echo "Converting $REC_MODEL to ONNX (paddle) format"
tar -xf ./models/paddle/$REC_MODEL.tar -C ./models/paddle
paddle2onnx --model_dir ./models/paddle/$REC_MODEL \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./models/paddle_onnx/$REC_MODEL.onnx \
--opset_version 11 \
--enable_onnx_checker True

# echo "Converting $CLS_MODEL to ONNX (paddle) format"
# tar -xf ./models/paddle/$CLS_MODEL.tar -C ./models/paddle
# paddle2onnx --model_dir ./models/paddle/$CLS_MODEL \
# --model_filename inference.pdmodel \
# --params_filename inference.pdiparams \
# --save_file ./models/paddle_onnx/$CLS_MODEL.onnx \
# --opset_version 11 \
# --enable_onnx_checker True


# ### Convert to RapidOCR onnx
# echo "Converting $DET_MODEL to ONNX (RapidOCR) format"
# paddleocr_convert -p ./models/paddle/$DET_MODEL.tar \
#                   -o ./models/rapid_onnx/$DET_MODEL.onnx

# echo "Converting $REC_MODEL to ONNX (RapidOCR) format"
# paddleocr_convert -p ./models/paddle/$REC_MODEL.tar \
#                   -o ./models/rapid_onnx/$REC_MODEL.onnx
#                   -txt_patth ./models/paddle/$REC_TXT

# echo "Converting $CLS_MODEL to ONNX (RapidOCR) format"
# paddleocr_convert -p ./models/paddle/$CLS_MODEL.tar \
#                   -o ./models/rapid_onnx/$CLS_MODEL.onnx