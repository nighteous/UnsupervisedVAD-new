import os

path = './Outputs/AE/BNBefore/'
x = os.listdir(path)

for i in x:
    
    f = open(path + i, 'r')
    lines = f.readlines()

    if len(lines) < 100:
        continue


    mean = 0

    for line in lines:
        if "Best" in line:
            z = line.split()
            mean += float(z[-1])
            
    mean = mean/5

    if mean > 60:
        print('cp {} {}'.format(path + i, "./Outputs/AE/Filtered"))
        os.system('cp {} {}'.format(path + i, "./Outputs/AE/Filtered"))