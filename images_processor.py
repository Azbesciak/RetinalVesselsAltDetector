from image_processing import process
from convolution_network import LearnData
from utils import Load, TEST_PATH

if __name__ == '__main__':
    data = LearnData(TEST_PATH)
    data.load_all()
    for img in data.original.images:
        res = process(img.image)
        Load.save(TEST_PATH + "/imageproc/" + img.get_file_name(), res)
