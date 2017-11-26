package com.github.neuralnetworks.calculation;

import org.junit.Test;

import com.github.neuralnetworks.architecture.Layer;
import com.github.neuralnetworks.architecture.NeuralNetworkImpl;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.calculation.BreadthFirstOrderStrategy;
import com.github.neuralnetworks.calculation.LayerOrderStrategy;
import com.github.neuralnetworks.test.AbstractTest;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by chass on 02.12.14.
 */
public class BreadthFirstOrderStrategyTests extends AbstractTest {

    @Test
    public void testLayerOrderStrategy()
    {
        // MLP
        NeuralNetworkImpl mlp = NNFactory.mlp(new int[]{3, 4, 5}, true);

        List<LayerOrderStrategy.ConnectionCandidate> ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
        assertEquals(4, ccc.size(), 0);
        Layer l = mlp.getOutputLayer();
        assertTrue(ccc.get(0).connection == l.getConnections().get(0));
        assertTrue(ccc.get(1).connection == l.getConnections().get(1));

        l = l.getConnections().get(0).getInputLayer();
        assertTrue(ccc.get(2).connection == l.getConnections().get(0));
        assertTrue(ccc.get(3).connection == l.getConnections().get(1));

        // Simple MLP
        mlp = NNFactory.mlp(new int[] { 3, 4 }, true);

        ccc = new BreadthFirstOrderStrategy(mlp, mlp.getOutputLayer()).order();
        assertEquals(2, ccc.size(), 0);
        l = mlp.getOutputLayer();
        assertTrue(ccc.get(0).connection == l.getConnections().get(0));
        assertTrue(ccc.get(1).connection == l.getConnections().get(1));

        // CNN
        NeuralNetworkImpl cnn = NNFactory.convNN(new int[][] { { 3, 3, 2 }, { 2, 2, 1, 1, 1, 0, 0 } }, true);

        ccc = new BreadthFirstOrderStrategy(cnn, cnn.getOutputLayer()).order();
        l = cnn.getOutputLayer();
        assertEquals(2, ccc.size(), 0);
        assertTrue(ccc.get(0).connection == l.getConnections().get(0));
        assertTrue(ccc.get(1).connection == l.getConnections().get(1));
    }
}
