package com.github.neuralnetworks.calculation.neuronfunctions;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import com.github.neuralnetworks.architecture.Connections;

/**
 * Singleton that stores all maxout winners for the backpropagation phase
 */
public class MaxoutWinners implements Serializable {

    private static final long serialVersionUID = 1L;

    private static MaxoutWinners singleton = new MaxoutWinners();

    private Map<Connections, Integer> startPositions;
    private int[] winners;
    private int batchSize;

    private MaxoutWinners() {
	startPositions = new HashMap<>();
	winners = new int[0];
    }

    public int getBatchSize() {
	return batchSize;
    }

    public void setBatchSize(int batchSize) {
	if (this.batchSize == 0) {
	    this.batchSize = batchSize;
	    this.winners = new int[0];
	    addConnections(startPositions.keySet().toArray(new Connections[0]));
	} else if (batchSize != this.batchSize) {
	    throw new IllegalArgumentException("This won't work");
	}
    }

    public int[] getWinners() {
	return winners;
    }

    public void addConnections(Connections... connections) {
	IntStream.range(0, connections.length).forEach(i -> {
	    startPositions.put(connections[i], winners.length);
	    winners = Arrays.copyOf(winners, winners.length + connections[i].getOutputUnitCount() * batchSize);
	});
    }

    public int[] getStartPositions(List<Connections> connections) {
	int[] result = new int[connections.size()];
	IntStream.range(0, connections.size()).forEach(i -> result[i] = startPositions.get(connections.get(i)));
	return result;
    }

    public static MaxoutWinners getInstance() {
	return singleton;
    }
}
