import gym
import time

def random_agent():
    env_name = 'BreakoutDeterministic-v4'
    env = gym.make(env_name)
    for i in range(10):
        env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            gameover = env.step(action)[2]
            if gameover:
                break
            time.sleep(1/30)

if __name__ == '__main__':
    random_agent()