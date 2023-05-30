from time import time
import tqdm
from setup import setup_game


def test_fps(iterations):
    game = setup_game()
    game.init()
    start = time()

    for i in tqdm.trange(iterations, leave=False):
        if game.is_episode_finished():
            game.new_episode()
        state = game.get_state()
        game.make_action([True, False, False])

    end = time()
    t = end - start

    print("Results:")
    print("Iterations:", iterations)
    print("time:", round(t, 3), "s")
    print("fps: ", round(iterations / t, 2))

    game.close()


if __name__ == '__main__':
    test_fps(10000)
