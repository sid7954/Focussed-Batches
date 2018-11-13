mkdir logs
python mnist_deep.py > logs/mnist.log
python mnist_deep_adv.py > logs/mnist_adv.log
python mnist_deep_rand_adv.py > logs/mnist_rand_adv.log
