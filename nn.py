import numpy as np
from tqdm import tqdm
import math

def sigmoid(num):
    return 1 / (1 + math.exp(-num))

def sigPrime(num):
    return sigmoid(num)*(1-sigmoid(num))

class Node(object):
    def __init__(self,initialValue,numPrevLayer=0):
        self.val = float(initialValue)
        self.weightsIn = np.zeros(numPrevLayer+1)

    def changeVal(self,newVal):
        self.val = float(newVal)

    def getVal(self):
        return self.val


class neuralNetwork(object):
    def __init__(self,numIn=0,numHidden=0,numOut=0):
        self.numIn = numIn
        self.numHidden = numHidden
        self.numOut = numOut

        self.inputLayer = []
        self.hiddenLayer = []
        self.outputLayer = []

        
    def initializeNetwork(self, initialFile):
        file = open(initialFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]
        #for d in data:
            #print(d)
            #print('-'*60)

        print('Initializing layers...')
        self.numIn = int(data[0][0])
        self.numHidden = int(data[0][1])
        self.numOut = int(data[0][2])

        for _ in range(self.numIn):
            n = Node(-1)
            self.inputLayer.append(n)
        
        for _ in range(self.numHidden):
            n = Node(-1,self.numIn)
            self.hiddenLayer.append(n)

        for _ in range(self.numOut):
            n = Node(-1,self.numHidden)
            self.outputLayer.append(n)

        # First weight is bias
        print('Initializing weights...')
        ind = 1
        for node in self.hiddenLayer:
            node.weightsIn = [float(w) for w in data[ind]]
            ind+=1

        for node in self.outputLayer:
            node.weightsIn = [float(w) for w in data[ind]]
            ind+=1
                
        #self.printWeights()



    def train(self,initFile,trainFile,epochs=50):
        self.initializeNetwork(initFile)
        LEARN_RATE = .1

        file = open(trainFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]

        numTrain = int(data[0][0])
        numFeatures = int(data[0][1])
        numOutput = int(data[0][2])

        x_data = [d[:numFeatures] for d in data[1:]]
        y_data = [d[numFeatures:] for d in data[1:]]
        self.printWeights()
        for _ in tqdm(range(epochs)):
            for x,y in zip(x_data,y_data):
                # Forward prop
                for ind,node in enumerate(self.inputLayer):
                    node.changeVal(x[ind])

                for ind,node in enumerate(self.hiddenLayer):
                    newNum = 0
                    newNum-=node.weightsIn[0]
                    for w in node.weightsIn[1:]:
                        #print(self.inputLayer[ind].getVal())
                        newNum+=(float(self.inputLayer[ind].getVal())*w)
                    node.changeVal(sigmoid(newNum))

                for ind,node in enumerate(self.outputLayer):
                    newNum = 0
                    newNum-=node.weightsIn[0]
                    for w in node.weightsIn[1:]:
                        #print(self.inputLayer[ind].getVal())
                        newNum+=(float(self.inputLayer[ind].getVal())*w)
                    node.changeVal(sigmoid(newNum))

                # Back prop

                # Compute loss
                lossOut = []
                for ind,node in enumerate(self.outputLayer):
                    currLoss = sigPrime(node.getVal()) * (float(y[0])-sigmoid(node.getVal()))
                    lossOut.append(currLoss)
                
                
                lossHidden = []
                currLossSig = sigPrime(-1)
                currLoss=0
                for n in self.outputLayer:
                    currLoss+=n.weightsIn[0]
                lossHidden.append(currLoss*currLossSig)

                for ind,node in enumerate(self.hiddenLayer):
                    currLossSig = sigPrime(node.getVal())
                    currLoss = 0
                    for ind2,n in enumerate(self.outputLayer):
                        #print(ind+1)
                        currLoss+=(n.weightsIn[ind+1]*lossOut[ind2])
                    lossHidden.append(currLoss*currLossSig)
                

                lossIn = []

                currLossSig = sigPrime(-1)
                currLoss=0
                for n in self.hiddenLayer:
                    currLoss+=n.weightsIn[0]
                lossIn.append(currLoss*currLossSig)


                for ind,node in enumerate(self.inputLayer):
                    currLossSig = sigPrime(node.getVal())
                    currLoss = 0
                    for ind2,n in enumerate(self.hiddenLayer):
                        currLoss+=(n.weightsIn[ind+1]*lossHidden[ind2])
                    lossIn.append(currLoss*currLossSig)

                # Update weights
                node.weightsIn[0]+=LEARN_RATE*sigmoid(-1)*lossIn[0]
                for ind,node in enumerate(self.hiddenLayer):
                    for ind2,w in enumerate(node.weightsIn[1:]):
                        newVal = LEARN_RATE*sigmoid(self.inputLayer[ind2].getVal())*lossIn[ind2]
                        node.weightsIn[ind2+1] = w+newVal
                
                
                
                
            #print(self.hiddenLayer[0].weightsIn)
            print('loss is:',sum(lossHidden))+sum(lossOut)+sum(lossIn))
            #print('sig:',lossIn[1])


        #self.printWeights()

    def predict(self,testFile):
        file = open(testFile,'r')
        lines = file.readlines()
        data = [l.split() for l in lines]

        numTest = int(data[0][0])
        numFeatures = int(data[0][1])
        numOutput = int(data[0][2])

        x_data = [d[:numFeatures] for d in data[1:]]
        y_data = [d[numFeatures:] for d in data[1:]]

        numTests = numTest
        numRight = 0

        for x,y in zip(x_data,y_data):
            # Forward prop
            for ind,node in enumerate(self.inputLayer):
                node.changeVal(x[ind])

            for ind,node in enumerate(self.hiddenLayer):
                newNum = 0
                newNum-=node.weightsIn[0]
                for w in node.weightsIn[1:]:
                    #print(self.inputLayer[ind].getVal())
                    newNum+=(float(self.inputLayer[ind].getVal())*w)
                node.changeVal(sigmoid(newNum))

            for ind,node in enumerate(self.outputLayer):
                newNum = 0
                newNum-=node.weightsIn[0]
                for w in node.weightsIn[1:]:
                    #print(self.inputLayer[ind].getVal())
                    newNum+=(float(self.inputLayer[ind].getVal())*w)
                node.changeVal(sigmoid(newNum))

            predictions = [str(round(n.getVal())) for n in self.outputLayer]

            #print('Real:',str(y))
            #print('Predictions:',str(predictions))
            if(str(y)==str(predictions)):
                print('YES')
                numRight+=1
            else:
                print('NO')
        print('Acc:',float(numRight)/float(numTest))

        

    def printWeights(self):
        #for node in self.inputLayer:
            #for w in node.weightsIn:
                #print(w,end=',')
            #print()

        for node in self.hiddenLayer:
            for w in node.weightsIn:
                print(w,end=',')
            print()

        for node in self.outputLayer:
            for w in node.weightsIn:
                print(w,end=',')
            print()



if __name__ == '__main__':

    nn = neuralNetwork(1,2,1)
    nn.train('./initialNetwork.txt','./trainFile.txt',epochs=50)
    nn.predict('./trainFile.txt')









