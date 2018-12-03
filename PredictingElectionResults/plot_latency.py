import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import csv
import numpy as np
from scipy import stats
import math

def main(args):
    colors = {'OpenFaaS regular': ['#003366','#ADD8E6'], 'OpenFaaS SCONE': ['#FF8C00','#FF4500'], 'VM regular': ['#006400','#90EE90'], 'VM SCONE': ['#FF6347','#FF0000']}
    latency_dict = dict()
    data = csv.reader(args.input, delimiter=",")
    next(data, None)

    # Store latency data
    for row in data:
        label = row[0]
        if label not in latency_dict:
            latency_dict[label] = {}
        msg_id = int(row[1])
        latency = float(row[2])
        if msg_id not in latency_dict[label]:
            latency_dict[label][msg_id] = {'latency':[]}
        latency_dict[label][msg_id]['latency'].append(latency)

    # Calculate metrics
    for label in latency_dict:
        msg_ids = sorted(latency_dict[label].keys())
        for msg_id in msg_ids:
            latency_arr = np.array(latency_dict[label][msg_id]['latency'])
            n, min_max, mean, var, skew, kurt = stats.describe(latency_arr)
            std = math.sqrt(var)
            R = stats.norm.interval(0.95,loc=mean,scale=std/math.sqrt(len(latency_arr)))
            if 'latency_means' not in latency_dict[label]:
                latency_dict[label]['latency_means'] = list()
            latency_dict[label]['latency_means'].append(mean)
            if 'error_height' not in latency_dict[label]:
                latency_dict[label]['error_height'] = list()
            latency_dict[label]['error_height'].append(abs(mean - R[0]))

        plt.plot(msg_ids, latency_dict[label]['latency_means'], 'k', color=colors[label][0])
        plt.fill_between(msg_ids, np.array(latency_dict[label]['latency_means'])-np.array(latency_dict[label]['error_height']), np.array(latency_dict[label]['latency_means'])+np.array(latency_dict[label]['error_height']), alpha=0.5, edgecolor=colors[label][0], facecolor=colors[label][1])

        print("Generated line for {:s}".format(label))
#        latencies = list(map(lambda x: float(latency_dict[label][x][0])/latency_dict[label][x][1], msg_ids))
#        plt.plot(msg_ids, latencies)
    plt.legend(['OpenFaaS regular', 'OpenFaaS SCONE', 'VM regular', 'VM SCONE'], loc='upper left')
    plt.xlabel("msg_id")
    plt.ylabel("latency (s)")
    plt.title("Latency for messages produced in\na single burst of 100 messages (95% CI)")
    plt.savefig(args.output,bbox_inches='tight')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', '-i', help='input filename', required=True, type=argparse.FileType('r'))
    arg_parser.add_argument('--output', '-o', help='output filename', required=True, type=str)
    main(arg_parser.parse_args())
