#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

"""
resnet模型by pytorch
"""
import sys
import torch.nn as nn
from torch.autograd import Variable
from data_processing_v2 import *
from torch.utils import data
import torchvision.models as models
import torch.cuda

def train(backup_file,test_dir='flowers_test'):
    gpu = 0  ## no gpu default
    
    train_dataset, test_dataset = load_train_dataset(), load_test_dataset(test_dir)


    # Image Preprocessing
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=16,
                                              shuffle=True)


    #pretrained=True就可以使用预训练的模型
    resnet18 = models.resnet18(num_classes=4)
    #resnet18.fc = torch.nn.Linear(512, 4)
    #resnet18.avgpool = nn.AvgPool2d(8, stride=1)
    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
        gpu = 1
    if not backup_file == None:
        try:
            resnet18.load_state_dict(torch.load(backup_file))
        except BaseException as e:
            print("the backup can't be found.")


    if 1 == gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    lr = 0.00001
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)

    # Training
    epoch_times = 100
    epoch_loss_record = []
    for epoch in range(epoch_times):
        for i, (images, labels) in enumerate(train_loader):
            if 1 == gpu:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images)
                labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch + 1, epoch_times, i + 1, loss.data[0]))

        # Decaying Learning Rate
        if (epoch + 1) % 20 == 0 and lr >3e-6:
            lr /= 1.3
            optimizer = torch.optim.Adam(resnet18.parameters(), lr=lr)
            print("the learning rate:" + ' ' + str(lr))

        if (epoch + 1) % 10 == 0:
            torch.save(resnet18.state_dict(), 'backup/resnet18_'+ str(epoch+1) +'.pkl')
            print('resnet18_'+ str(epoch+1) +'.pkl' + " has been saved")

        epoch_loss_record.append(loss.data[0])


        # Test
    correct = 0
    total = 0
    for images, labels in test_loader:

        if 1 == gpu:
            images = Variable(images).cuda()
            outputs = resnet18(images).cuda()
        else:
            images = Variable(images)
            outputs = resnet18(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=(predicted.cpu().int()==labels.cpu().int()).sum()

    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(resnet18.state_dict(), 'backup/resnet18_final.pkl')

    return 0

def test(backup_file,test_dir='flowers_test'):
    test_gpu = 0
    
    test_dataset = load_test_dataset(test_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=16,
                                              shuffle=True)

    resnet18 = models.resnet18(num_classes=4)
    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
        test_gpu = 1
    if not backup_file == None:
        try:
            resnet18.load_state_dict(torch.load(backup_file))
        except BaseException as e:
            print("the backup can't be found.")

    correct = 0
    total = 0
    waiting = 0
    for images, labels in test_loader:
        if 1 == test_gpu:
            images = Variable(images).cuda()
            outputs = resnet18(images).cuda()
        else:
            images = Variable(images)
            outputs = resnet18(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #correct += (predicted == labels).sum()
        correct +=(predicted.cpu().int()==labels.cpu().int()).sum()
        waiting += 1
        if waiting % 15 == 0:
            print("===============waiting(" + str(waiting*16)+' images' + ")===================")

    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    return 0


def main(argv):

    if argv[1] not in ['test','train']:
        print("the second argument is not valid\n" + "your input " + argv[1] + 'is not test or train')
    else:
        task = argv[1]
        backup_file = argv[2]
        if len(argv)>3:
            test_dir = argv[3]

    if task == 'train':
        if len(argv) == 4:
            train(backup_file,test_dir)
        elif len(argv) == 3:
            train(backup_file)
    elif task == 'test':
        if len(argv) == 4:
            test(backup_file,test_dir)
        elif len(argv) == 3:
            test(backup_file)

    return 0

if __name__ == '__main__':
    #print(sys.argv)
    main(sys.argv)


