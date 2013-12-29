package com.github.neuralnetworks.calculation.neuronfunctions;

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
     * output kernels count
     */
    protected int outputKernels;

    /**
     * output columns * output rows
     */
    protected int featureMapLength;

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
    @Local
    protected float[] weights;
    
    /**
     * weights for single feature map
     */
    protected int featureMapWeights;

    /**
     * input offset for each feature map in respect to the start index
     */
    @Local
    protected int[] featureMapOffsets;

    /**
     * current connection
     */
    protected Conv2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	Conv2DConnection c = (Conv2DConnection) connections.keySet().iterator().next();

	this.init(c, connections.values().iterator().next(), output);

	ConvGridLayer tl = (ConvGridLayer) targetLayer;

	this.execute(tl.getFeatureMapLength());
    }

    /**
     * TODO
     * Converts connection, input and output data to one dimensional arrays (because of the Aparapi limitations)
     */
    protected void init(Conv2DConnection c, Matrix input, Matrix output) {
	if (current != c) {
	    current = c;

	    ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	    this.input = input.getElements();
	    this.output = output.getElements();
	    this.weights = c.getWeights();
	    this.inputColumns = inputLayer.getFeatureMapColumns();
	    this.outputColumns = outputLayer.getFeatureMapColumns();
	    this.outputKernels = outputLayer.getFilters();
	    this.featureMapLength = outputLayer.getFeatureMapLength();
	    this.featureMapWeights = c.getWeights().length / outputLayer.getFilters();
	    this.featureMapOffsets = new int[featureMapWeights];
	    
	    int offset = 0;
	    
	    for (int i = 0; i < inputLayer.getFilters(); i++) {
		for (int j = 0; j < c.getKernelRows(); j++) {
		    for (int k = 0; k < c.getKernelColumns(); k++) {
			featureMapOffsets[offset++] = i * inputLayer.getFeatureMapLength() + j * inputLayer.getFeatureMapColumns() + k;
		    }
		}
	    }
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    }

    @Override
    public void run() {
	int id = getGlobalId();

	// calculate sum based on feature map offsets and feature map weights
	float sum = 0;
	int fmw = featureMapWeights;
	int fml = featureMapLength;
	int ok = outputKernels;

	for (int k = 0; k < ok; k++) {
	    for (int i = 0, j = fmw * k; i < fmw; i++, j++) {
		sum += input[id + featureMapOffsets[i]] * weights[j];
	    }

	    output[k * fml + id] = sum;

	    after();
	}
    }

    /**
     * this is called after the convolution
     */
    protected void after() {
    }
}
