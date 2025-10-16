import argparse
from src.aeonforge import demo

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('+demo', dest='demo', action='store_true'); a=p.parse_args()
    print('Running demo...' if a.demo else 'Use `+demo=true`')
    if a.demo: demo.run()
