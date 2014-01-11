package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.ConvGridLayer;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;
import com.github.neuralnetworks.util.Environment;

/**
 * Base Aparapi connection calculator for subsampling layers.
 * 
 * !!! IMPORTANT !!!
 * Aparapi only works one-dimensional arrays of primitive data types can only call member methods of the Kernel class itself.
 */
public abstract class AparapiSubsampling2D extends Kernel implements ConnectionCalculator {

    private static final long serialVersionUID = 8931101094464503687L;
    
    /**
     * input feature map columns
     */
    protected int inputFeatureMapColumns;

    /**
     * Length of the input image (rows * cols)
     */
    protected int inputFeatureMapLength;
    
    /**
     * output feature map columns
     */
    protected int outputFeatureMapColumns;

    /**
     * Length of the output image (rows * cols)
     */
    protected int outputFeatureMapLength;

    /**
     * input samples count
     */
    protected int inputOutputSamples;

    /**
     * subsampling region rows
     */
    protected int subsamplingRows;
    
    /**
     * subsampling region columns
     */
    protected int subsamplingCols;

    /**
     * Length of the subsampling region (subsampling rows *  subsampling cols)
     */
    protected int regionLength;

    /**
     * offset from start when mapping input to output
     */
    protected int ioRowsOffset;

    /**
     * offset from start when mapping input to output
     */
    protected int ioColumnsOffset;

    /**
     * input data
     */
    protected float[] input;

    /**
     * output
     */
    protected float[] output;

    /**
     * Contains the offset in the input array for each cell of the current region. The offset is calculated in respect to the first cell of the region
     */
    //@Local TODO
    protected int[] featureMapOffsets;

    protected Subsampling2DConnection current;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	if (connections.size() > 0) {
	    Subsampling2DConnection c = (Subsampling2DConnection) connections.keySet().iterator().next();
	    if (targetLayer == c.getOutputLayer()) {
		init(c, connections.values().iterator().next(), output);
	    } else {
		init(c, output, connections.values().iterator().next());
	    }

	    // the code is executed with as many kernels as the output layer neurons count
	    execute(c.getOutputLayer().getNeuronCount());
	}
    }

    /**
     * Populates featureMapOffsets and outputInputIndexes
     * @param c
     * @param input
     * @param output
     */
    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	this.inputOutputSamples = output.getColumns();

	if (c != current) {
	    current = c;

	    ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	    ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	    this.input = input.getElements();
	    this.output = output.getElements();
	    this.inputFeatureMapColumns = inputLayer.getFeatureMapColumns();
	    this.inputFeatureMapLength = inputLayer.getFeatureMapLength();
	    this.outputFeatureMapColumns = outputLayer.getFeatureMapColumns();
	    this.outputFeatureMapLength = outputLayer.getFeatureMapLength();
	    this.subsamplingRows = c.getSubsamplingRegionRows();
	    this.subsamplingCols = c.getSubsamplingRegionCols();
	    this.regionLength = subsamplingRows * subsamplingCols;
	    this.ioRowsOffset = (input.getRows() % subsamplingRows) / 2;
	    this.ioColumnsOffset = (input.getColumns() % subsamplingCols) / 2;
	    this.featureMapOffsets = new int[regionLength];

	    for (int i = 0, j = 0; j < subsamplingRows; j++) {
		for (int k = 0; k < subsamplingCols; k++, i++) {
		    featureMapOffsets[i] = j * inputLayer.getFeatureMapColumns() + k;
		}
	    }
	}

	setExecutionMode(Environment.getInstance().getExecutionMode());
    }

    /* (non-Javadoc)
     * @see com.amd.aparapi.Kernel#run()
     * currentValues array is populated here. The values from this array are the ones that take part in the pooling in currentValuesUpdated
     */
    @Override
    public void run() {
	int id = getGlobalId();

	// get current offset
	int fmOffset = id % outputFeatureMapLength;

	pool((id / outputFeatureMapLength) * inputFeatureMapLength + (ioRowsOffset + (fmOffset / outputFeatureMapColumns) * subsamplingRows) * inputFeatureMapColumns + (fmOffset % outputFeatureMapColumns) * subsamplingCols);
    }

    /**
     * This is where the subsampling happens
     */
    protected void pool(int inputStartIndex) {
    }
}
