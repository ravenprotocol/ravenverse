from ravpy.initialize import initialize
from ravpy.distributed.participate import participate

if __name__ == '__main__':
    ravenverse_token = '<ravenverse_token>'
    username = '1'

    initialize(ravenverse_token, username)
    participate()