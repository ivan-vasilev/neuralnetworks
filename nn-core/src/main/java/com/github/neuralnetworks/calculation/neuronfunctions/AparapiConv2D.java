package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Conv2DConnection;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * Convolution connection calculator (2d)
 */
public class AparapiConv2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    protected int inputColumns;
    protected int outputColumns;
    protected int outputLength;
    protected float[] input;
    protected float[] output;
    protected float[] featureMapWeights;
    protected int[] featureMapOffsets;
    protected int[] featureMapStartIndexes;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = (Conv2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	this.execute(targetLayer.getNeuronCount());
    }

    protected void init(Conv2DConnection c, Matrix input, Matrix output) {
	ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	this.input = input.getElements();
	this.output = output.getElements();
	this.inputColumns = inputLayer.getColumns();
	this.outputColumns = outputLayer.getColumns();
	this.outputLength = outputLayer.getColumns() * outputLayer.getRows();
	this.featureMapStartIndexes = new int[c.getFilters().size() + 1];

	List<Integer> featureMapOffsets = new ArrayList<>();
	List<Float> featureMapWeights = new ArrayList<>();
	int offset = 0;

	for (int i = 0; i < c.getFilters().size(); i++) {
	    Matrix fm = c.getFilters().get(i);
	    featureMapStartIndexes[i] = offset;
	    for (int j = 0; j < fm.getRows(); j++) {
		for (int k = 0; k < fm.getColumns(); k++) {
		    featureMapWeights.add(fm.getElements()[k]);
		    featureMapOffsets.add(i * inputLayer.getRows() * inputLayer.getColumns() + j * inputLayer.getColumns() + k);
		    offset++;
		}
	    }
	}

	featureMapStartIndexes[c.getFilters().size()] = featureMapOffsets.size() - 1;

	if (this.featureMapOffsets == null || this.featureMapOffsets.length != featureMapOffsets.size()) {
	    this.featureMapOffsets = new int[featureMapOffsets.size()];
	}
	
	if (this.featureMapWeights == null || this.featureMapWeights.length != featureMapOffsets.size()) {
	    this.featureMapWeights = new float[featureMapOffsets.size()];
	}

	for (int i = 0; i < featureMapOffsets.size(); i++) {
	    this.featureMapOffsets[i] = featureMapOffsets.get(i);
	    this.featureMapWeights[i] = featureMapWeights.get(i);
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    }

    @Override
    public void run() {
	int id = getGlobalId();

	// get feature map data
	int featureMap = id / outputLength;
	int featureMapStartIndex = featureMapStartIndexes[featureMap];
	int featureMapEndIndex = featureMapStartIndexes[featureMap + 1];

	// get input index
	int currentImageMapIndex = id % outputLength;
	int inputIndex = (currentImageMapIndex / outputColumns) * inputColumns + currentImageMapIndex % outputColumns;

	float sum = 0;
	for (; featureMapStartIndex < featureMapEndIndex; featureMapStartIndex++) {
	    sum += input[inputIndex + featureMapOffsets[featureMapStartIndex]] * featureMapWeights[featureMapStartIndex];
	}

	output[id] = sum;

	after();
    }

    /**
     * this is called after the convolution
     */
    protected void after() {
    }
}
