import json
import math

my_latitude   = 12.9611159
my_longitude  = 77.6362214
earth_radiues = 6371


try:
	with open('friends.json') as json_data:
	    d = json.load(json_data)

	friends = []

	results = []

	for i in range(0, len(d)):

		frnd_latitude  = d[i]['latitude']
		frnd_longitude = d[i]['longitude']

		if type(d[i]['_id']) == int and type(d[i]['latitude']) != str and type(d[i]['longitude']) != str:

			delta_sigma = math.acos(math.sin(my_latitude)*math.sin(frnd_latitude) + math.cos(my_latitude)*math.cos(frnd_latitude)*math.cos(frnd_longitude - my_longitude))

			dist = earth_radiues * delta_sigma


			item = {
					'Id'		: d[i]['_id'],
					'Latitude'  : d[i]['latitude'],
					'Longitude' : d[i]['longitude'],
					'Name'		: str(d[i]['name']),
					'Distance'	: dist
					}

			if dist <= 100:
				results.append(item)

			friends.append(item)

	results = sorted(results, key=lambda k : k['Id'])

	for i in range(0, len(results)):
		print results[i]['Name'], results[i]['Id']


except ValueError:
	print 'Error reading the json file'

