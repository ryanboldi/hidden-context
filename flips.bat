echo "starting"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.1
echo "0.1 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.2
echo "0.2 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.3
echo "0.3 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.4
echo "0.4 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.45
echo "0.45 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.5
echo "0.5 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.55
echo "0.55 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.6
echo "0.6 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.75
echo "0.75 done"
python -m hidden_context.synthetic_experiments --env 1d_identity --batch_size 2048 --lr 0.001 --num_iterations 1000 --flips 0.9
echo "0.9 done"
echo "Done"