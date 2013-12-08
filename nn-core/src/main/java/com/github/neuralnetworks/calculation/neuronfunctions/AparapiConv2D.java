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
 * This connection accept as input a single training example (as opposed to the weighted sum which works with multiple).
 *
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 * 
 * 
 */
public class AparapiConv2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;

    /**
     * input column count (columns of image for example)
     */
    protected int inputColumns;

    /**
     * output column count (columns of image for example)
     */
    protected int outputColumns;

    /**
     * output columns * output rows
     */
    protected int outputLength;

    /**
     * input
     */
    protected float[] input;

    /**
     * output
     */
    protected float[] output;

    /**
     * combined feature weights of all feature maps
     */
    protected float[] featureMapWeights;

    /**
     * input offset for each feature map in respect to the start index
     */
    protected int[] featureMapOffsets;

    /**
     * feature map start indexes in the featureMapOffsets array
     */
    protected int[] featureMapStartIndexes;

    /**
     * current connection
     */
    protected Conv2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = (Conv2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	this.execute(targetLayer.getNeuronCount());
    }

    /**
     * Converts connection, input and output data to one dimensional arrays (because of the Aparapi limitations)
     */
    protected void init(Conv2DConnection c, Matrix input, Matrix output) {
	if (current != c) {
	    current = c;

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

	// calculate sum based on feature map offsets and feature map weights
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
