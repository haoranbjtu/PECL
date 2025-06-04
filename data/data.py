class Data(object):
    def __init__(self, conf, training, test):
    # def __init__(self, conf, training, test, training_time, test_time):
        self.config = conf
        self.training_data = training
        # self.training_time = training_time
        self.test_data = test #can also be validation set if the input is for validation
        # self.test_time = test_time


