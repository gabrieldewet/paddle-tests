# Download angle classifier model using curl
curl -o ./models/paddle/ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

# Download detection model using curl
curl -o ./models/paddle/Multilingual_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar

# Download recognition model and text file for appropriate language using curl
curl -o ./models/paddle/latin_PP-OCRv3_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar
curl -o ./models/paddle/latin_dict.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/ppocr/utils/dict/latin_dict.txt