from typing import List
import digitalocean
from digitalocean import Volume
manager = digitalocean.Manager(
    token='21dc15072955888e339029c163470a73ebb9e2ae69082c4d58edac6c73488761'
)
my_volumes: List[Volume] = manager.get_all_volumes()
# prin÷t(my_volumes)
for vol in my_volumes:
    print(vol)
    try:
        vol.destroy()
    except Exception as identifier:
        print("Couldn't remove it.")
