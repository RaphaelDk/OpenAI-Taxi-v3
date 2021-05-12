chmod +x main.py

./main.py [agent] [iterations]

Sans le .env à la fin du make, l'algo s'arrête à 200 itérations

The filled square represents the taxi, which is yellow without a passenger and green with a passenger.

The pipe ("|") represents a wall which the taxi cannot cross.

R, G, Y, B are the possible pickup and destination locations. The blue letter represents the current passenger pick-up location, and the purple letter is the current destination.

0 = south
1 = north
2 = east
3 = west
4 = pickup
5 = dropoff

Inspiré de : https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
Understanding Q learning : https://medium.com/dataseries/understanding-the-idea-behind-q-learning-63c666c8a8a2
