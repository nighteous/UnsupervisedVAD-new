import os 

models = ["2048,1024,512,256,512,1024,2048", "2048,1024,512,256,128,248,512,1024,2048", "2048,1024,512,256,128,64,128,256,512,1024,2048", "2048,1024,512,128,64,32,64,128,256,512,1024,2048"]

for i in models:
    command_train = "python train_AE.py --arch {}".format(i)
    command_test = "python test_AE.py --arch {}".format(i)

    print(command_train)
    os.system(command_train)
    print(command_test)
    os.system(command_test)