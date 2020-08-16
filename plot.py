import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

	# plot losses
	if os.path.exists('train_logs.txt') and os.path.exists('val_logs.txt'):
		train_losses = []
		val_losses = []

		train = open('train_logs.txt','r')
		train_all = train.readlines()
		for line in train_all:
			train_losses.append(float(line.strip("\n").split(',')[1].split(':')[1].strip()))
		print(train_losses)

		val = open('val_logs.txt','r')
		val_all = val.readlines()
		for line in val_all:
			val_losses.append(float(line.strip("\n").split(',')[1].split(':')[1].strip()))
		print(val_losses)

		plt.figure()
		plt.title('Losses', fontsize=20)
		plt.xlabel('epoch', fontsize=15)
		plt.ylabel('loss', fontsize=15)
		plt.plot(train_losses, linewidth=2,label='train loss')
		plt.plot(val_losses, linewidth=2, label='val loss')
		plt.legend()
		plt.savefig('losses')
		plt.show()
		plt.clf()

	# plot accuracy
	if os.path.exists('val_accuracy.txt'):
		ious = list()
		IOUs = open("val_accuracy.txt",'r').readlines()
		for IOU in IOUs:
			ious.append(float(IOU.strip('\n')))
		print(ious)
		plt.figure()
		plt.title('Val Average Accuracy', fontsize=20)
		plt.xlabel('epoch', fontsize=15)
		plt.ylabel('Accuracy', fontsize=15)
		plt.plot(ious, linewidth=2)
		plt.savefig('Accuracys')
		plt.show()
		plt.clf()

	# plot IOU
	if os.path.exists('val_IOU.txt'):
		ious = list()
		IOUs = open("val_IOU.txt",'r').readlines()
		for IOU in IOUs:
			ious.append(float(IOU.strip('\n')))
		print(ious)
		plt.figure()
		plt.title('Val IOUs', fontsize=20)
		plt.xlabel('epoch', fontsize=15)
		plt.ylabel('IOU', fontsize=15)
		plt.plot(ious, linewidth=2)
		plt.savefig('IOUs')
		plt.show()
		plt.clf()



