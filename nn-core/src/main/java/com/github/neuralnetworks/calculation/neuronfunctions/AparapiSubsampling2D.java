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
    protected final int inputFeatureMapColumns;

    /**
     * Length of the input image (rows * cols)
     */
    protected final int inputFeatureMapLength;
    
    /**
     * output feature map columns
     */
    protected final int outputFeatureMapColumns;

    /**
     * Length of the output image (rows * cols)
     */
    protected final int outputFeatureMapLength;

    /**
     * input samples count
     */
    protected final int miniBatchSize;

    /**
     * subsampling region rows
     */
    protected final int subsamplingRows;
    
    /**
     * subsampling region columns
     */
    protected final int subsamplingCols;

    /**
     * Length of the subsampling region (subsampling rows *  subsampling cols)
     */
    protected final int regionLength;

    /**
     * offset from start when mapping input to output
     */
    protected final int ioRowsOffset;

    /**
     * offset from start when mapping input to output
     */
    protected final int ioColumnsOffset;

    /**
     * Contains the offset in the input array for each cell of the current region. The offset is calculated in respect to the first cell of the region
     */
    //@Local TODO
    protected final int[] featureMapOffsets;

    /**
     * input data
     */
    protected float[] input;

    /**
     * output
     */
    protected float[] output;

    public AparapiSubsampling2D(Subsampling2DConnection c, int miniBatchSize) {
	this.miniBatchSize = miniBatchSize;
	ConvGridLayer inputLayer = (ConvGridLayer) c.getInputLayer();
	ConvGridLayer outputLayer = (ConvGridLayer) c.getOutputLayer();
	this.inputFeatureMapColumns = inputLayer.getFeatureMapColumns();
	this.inputFeatureMapLength = inputLayer.getFeatureMapLength();
	this.outputFeatureMapColumns = outputLayer.getFeatureMapColumns();
	this.outputFeatureMapLength = outputLayer.getFeatureMapLength();
	this.subsamplingRows = c.getSubsamplingRegionRows();
	this.subsamplingCols = c.getSubsamplingRegionCols();
	this.regionLength = c.getSubsamplingRegionLength();
	this.ioRowsOffset = (inputLayer.getFeatureMapRows() % subsamplingRows) / 2;
	this.ioColumnsOffset = (inputLayer.getFeatureMapColumns() % subsamplingCols) / 2;
	this.featureMapOffsets = new int[regionLength];

	for (int i = 0, j = 0; j < subsamplingRows; j++) {
	    for (int k = 0; k < subsamplingCols; k++, i++) {
		featureMapOffsets[i] = j * inputLayer.getFeatureMapColumns() + k;
	    }
	}
    }

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	if (connections.size() > 0) {
	    Subsampling2DConnection c = (Subsampling2DConnection) connections.keySet().iterator().next();
	    if (targetLayer == c.getOutputLayer()) {
		init(c, connections.values().iterator().next(), output);
		Environment.getInstance().getExecutionStrategy().execute(this, output.getRows());
	    } else {
		Matrix m = connections.values().iterator().next();
		init(c, output, m);
		Environment.getInstance().getExecutionStrategy().execute(this, m.getRows());
	    }

	}
    }

    /**
     * Populates featureMapOffsets and outputInputIndexes
     * @param c
     * @param input
     * @param output
     */
    protected void init(Subsampling2DConnection c, Matrix input, Matrix output) {
	this.input = input.getElements();
	this.output = output.getElements();
    }

    /* (non-Javadoc)
     * @see com.amd.aparapi.Kernel#run()
     * input start index is calculated and passed to the pooling method
     */
    @Override
    public void run() {
	int id = getGlobalId();

	// get current offset
	int fmOffset = id % outputFeatureMapLength;

	pool((id / outputFeatureMapLength) * inputFeatureMapLength + (ioRowsOffset + (fmOffset / outputFeatureMapColumns) * subsamplingRows) * inputFeatureMapColumns + ioColumnsOffset + (fmOffset % outputFeatureMapColumns) * subsamplingCols);
    }

    /**
     * This is where the subsampling happens
     */
    protected void pool(int inputStartIndex) {
    }
}
