package com.github.neuralnetworks.calculation.neuronfunctions;

import java.util.SortedMap;

import com.github.neuralnetworks.architecture.Connections;
import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.Matrix;
import com.github.neuralnetworks.architecture.Subsampling2DConnection;
import com.github.neuralnetworks.calculation.ConnectionCalculator;

/**
 * Max pooling
 */
public class AparapiMaxPooling2D implements ConnectionCalculator {

    private static final long serialVersionUID = 8165829315701496713L;

    private ConnectionCalculator cc;

    @Override
    public void calculate(SortedMap<Connections, Matrix> connections, Matrix output, Layer targetLayer) {
	if (cc == null) {
	    cc = new AparapiMaxPooling2DCC((Subsampling2DConnection) connections.keySet().iterator().next(), output.getColumns());
	}

	cc.calculate(connections, output, targetLayer);
    }

    public static class AparapiMaxPooling2DCC extends AparapiSubsampling2D {

	private static final long serialVersionUID = -2393526660090301257L;

	public AparapiMaxPooling2DCC(Subsampling2DConnection c, int miniBatchSize) {
	    super(c, miniBatchSize);
	}

	@Override
	protected void pool(int inputStartIndex) {
	    int rl = regionLength;
	    int miniBatch = miniBatchSize;
	    float max = 0;

	    for (int i = 0; i < miniBatch; i++) {
		max = input[(inputStartIndex + featureMapOffsets[0]) * miniBatch + i];
		for (int j = 1; j < rl; j++) {
		    max = max(input[(inputStartIndex + featureMapOffsets[j]) * miniBatch + i], max);
		}

		output[getGlobalId() * miniBatch + i] = max;
	    }
	}
    }
}
