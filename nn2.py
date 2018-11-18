import numpy as np
from tqdm import tqdm
import math

def sigmoid(num):
    return 1 / (1 + math.exp(-num))

def sigPrime(num):
    return sigmoid(num)*(1-sigmoid(num))

def sigInverse(num):
    return math.log(num/(1-num))

def printWeights(weights):
    for w in weights:
        for ww in w:
            print(ww,end=', ')
        print()

class Node(object):
    def __init__(self,initialValue):
        self.val = float(initialValue)
        self.sigVal = (sigmoid(initialValue))

    def changeVal(self,num):
        self.val = float(num)
        self.sigVal = (sigmoid(num))

class neuralNet(object):
    def __init__(self,initialFile):
        file = open(initialFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]
        print('Initializing layers...')
        numInput = int(data[0][0])
        numHidden = int(data[0][1])
        numOutput = int(data[0][2])

        self.inputLayer = [Node(-1) for _ in range(numInput+1)]
        self.hiddenLayer = [Node(-1) for _ in range(numHidden+1)]
        self.outputLayer = [Node(-1) for _ in range(numOutput)]
        
        self.weights1 = np.zeros(shape=(len(self.hiddenLayer)-1,len(self.inputLayer))) # minus is bc of the bias node which weights are not fed into
        self.weights2 = np.zeros(shape=(len(self.outputLayer),len(self.hiddenLayer))) # ouput has no bias nodes

        print('Network has',numInput+1,'input nodes. ',numHidden+1,'hidden nodes. ',numOutput,'output nodes.')
        print('Weights1 has shape:',self.weights1.shape,'Weights2 has shape:',self.weights2.shape)

        print('Initializing weights...')
        ind = 1
        for ind1,row in enumerate(self.weights1):
            for ind2,w in enumerate(row):
                self.weights1[ind1][ind2] = float(data[ind][ind2])
            ind+=1

        for ind1,row in enumerate(self.weights2):
            for ind2,w in enumerate(row):
                self.weights2[ind1][ind2] = float(data[ind][ind2])
            ind+=1



    def train(self,trainFile, epochs=50, alpha=0.1):
        LEARN_RATE = alpha

        file = open(trainFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]

        numTrain = int(data[0][0])
        numFeatures = int(data[0][1])
        numOutput = int(data[0][2])

        x_data = [d[:numFeatures] for d in data[1:]]
        y_data = [d[numFeatures:] for d in data[1:]]

        for _ in tqdm(range(epochs)):
            totalLoss = 0
            for x,y in zip(x_data,y_data):
                
                # Forward propogation
                for ind,node in enumerate(self.inputLayer[1:]):
                    node.changeVal(float(x[ind]))

                for ind,node in enumerate(self.hiddenLayer[1:]):
                    newValue = 0
                    for ind2,w in enumerate(self.weights1[ind]):
                        newValue+=(w*self.inputLayer[ind2].val)
                        #print('newVal=',newValue)
                    self.hiddenLayer[ind+1].changeVal(newValue)

                for ind,node in enumerate(self.outputLayer):
                    newValue = 0
                    for ind2,w in enumerate(self.weights2[ind]):
                        newValue+=(w*self.hiddenLayer[ind2].sigVal)
                        #print('newVal=',newValue)
                    self.outputLayer[ind].changeVal(newValue)

                
                #print('y is:',y[0],'pred is:',self.outputLayer[0].sigVal)


                # Back prop
                lossOut = []
                for ind,node in enumerate(self.outputLayer):
                    #print('Evaluating:',y[ind],' - ',node.sigVal)
                    currLoss = sigPrime(node.val)*(float(y[ind])-node.sigVal)
                    lossOut.append(currLoss)

                lossHidden = []
                for ind,node in enumerate(self.hiddenLayer):
                    currLossPrime = sigPrime(node.val)
                    currLoss = 0
                    for ind2,w in enumerate(self.weights2):
                        currLoss+=(w[ind]*lossOut[ind2])
                    lossHidden.append(currLoss*currLossPrime)

                lossIn = []
                for ind,node in enumerate(self.inputLayer):
                    currLossPrime = sigPrime(node.val)
                    currLoss = 0
                    for ind2,w in enumerate(self.weights1):
                        currLoss+=(w[ind]*lossHidden[ind2])
                    lossIn.append(currLoss*currLossPrime)

                totalLoss+=sum(lossIn)+sum(lossHidden)+sum(lossOut)
                
                for ind,w1 in enumerate(self.weights2):
                    for ind2,w2 in enumerate(w1):
                        self.weights2[ind][ind2] += LEARN_RATE*self.hiddenLayer[ind2].sigVal*lossOut[ind] 

                for ind,w1 in enumerate(self.weights1):
                    for ind2,w2 in enumerate(w1):
                        self.weights1[ind][ind2] += LEARN_RATE*self.inputLayer[ind2].val*lossHidden[ind] 

                #print('loss:',sum(lossIn)+sum(lossHidden)+sum(lossOut))
                #print('loss:',sum(lossIn))
            print('Loss:',totalLoss)



    def predict(self,testFile):
        file = open(testFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]

        numTest = int(data[0][0])
        numFeatures = int(data[0][1])
        numOutput = int(data[0][2])

        x_data = [d[:numFeatures] for d in data[1:]]
        y_data = [d[numFeatures:] for d in data[1:]]

        numTests = 0
        numRight = 0

        overall_accuracy = np.zeros(numOutput)
        precision = np.zeros(numOutput)
        recall =np.zeros(numOutput)
        F1 =np.zeros(numOutput)

        A=list(np.zeros(numOutput))
        B=list(np.zeros(numOutput))
        C=np.zeros(numOutput)
        D=np.zeros(numOutput)

        for x,y in zip(x_data,y_data):
            # Forward prop
            # Forward propogation
            for ind,node in enumerate(self.inputLayer[1:]):
                node.val = float(x[ind])

            for ind,node in enumerate(self.hiddenLayer[1:]):
                newValue = 0
                for ind2,w in enumerate(self.weights1[ind]):
                    newValue+=(w*self.inputLayer[ind2].val)
                    #print('newVal=',newValue)
                self.hiddenLayer[ind+1].val = sigmoid(newValue)

            for ind,node in enumerate(self.outputLayer):
                newValue = 0
                for ind2,w in enumerate(self.weights2[ind]):
                    newValue+=(w*self.hiddenLayer[ind2].val)
                    #print('newVal=',newValue)
                self.outputLayer[ind].val = sigmoid(newValue)

            
            predictions = [str(int(n.val/.5)) for n in self.outputLayer]

            #print('Real:',str(y))
            #print('Predictions:',str(predictions))

            for totalInd,p in enumerate(predictions):
                if(int(p)==int(y[ind])):
                    if(int(p)==1):
                        A[totalInd]+=1
                    else:
                        D[totalInd]+=1
                else:
                    if(int(p)==1):
                        B[totalInd]+=1
                    else:
                        C[totalInd]+=1
        overall_accuracy[totalInd] = ((A[totalInd]+D[totalInd])/(A[totalInd]+B[totalInd]+C[totalInd]+D[totalInd]))
        precision[totalInd] = ((A[totalInd])/(A[totalInd]+B[totalInd]))
        recall[totalInd] = ((A[totalInd])/(A[totalInd]+C[totalInd]))
        F1[totalInd] = ((2*precision[-1]*recall[-1])/(precision[-1]+recall[-1]))
        for ind,num in enumerate(overall_accuracy):
            print('Overall accuracy:',round(overall_accuracy[ind],3))
            print('Precision:',round(precision[ind],3))
            print('Recall:',round(recall[ind],3))
            print('F1:',round(F1[ind],3))


        #printWeights(self.weights2)





nn = neuralNet('./initialNetwork.txt')
nn.train('./trainFile.txt',epochs=10,alpha=0.1)
nn.predict('./testFile.txt')
nn = neuralNet('./finalNetworkSable.txt')
nn.train('./trainFile.txt',epochs=10,alpha=0.1)
nn.predict('./testFile.txt')

