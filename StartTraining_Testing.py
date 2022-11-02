import os 

models = ["1024,512,256,128,256,512,1024" ,"1024,512,256,128,64,128,256,512,1024", "1024,512,256,128,64,32,64,128,256,512,1024", "1024,512,256,128,64,32,16,32,64,128,256,512,1024"]
lr = [0.01 ,1e-4, 3e-4]
weightdecay = [0, 1e-4]
dropout = [0, 0.4, 0.6]
opti = ["Adam", "SGD"]

for arch in models:
    for l in lr:
        for opt in opti:
            for wd in weightdecay:
                for d in dropout:
                    command_train = "python train_AE_i3d.py --arch {} --lr {} --optimizer {} --weight-decay {} --dropout {}".format(arch, l, opt, wd, d)

                    print(command_train)
                    os.system(command_train)