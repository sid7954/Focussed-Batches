# python mnist_deep.py > logs/some
# tail -n 2 logs/some >> results
# python mnist_deep.py > logs/some
# tail -n 2 logs/some >> results
# python mnist_deep.py > logs/some
# tail -n 2 logs/some >> results

# python mnist_deep_adv.py > logs/some
# tail -n 2 logs/some >> results
# python mnist_deep_adv.py > logs/some
# tail -n 2 logs/some >> results
# python mnist_deep_adv.py > logs/some
# tail -n 2 logs/some >> results

echo "DEEP RAND" >> results
python mnist_deep_rand_adv.py > logs/some
tail -n 2 logs/some >> results
python mnist_deep_rand_adv.py > logs/some
tail -n 2 logs/some >> results
python mnist_deep_rand_adv.py > logs/some
tail -n 2 logs/some >> results
