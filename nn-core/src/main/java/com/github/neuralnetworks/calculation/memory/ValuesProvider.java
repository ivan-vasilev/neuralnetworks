package com.github.neuralnetworks.calculation.memory;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.neuralnetworks.util.Tensor;
import com.github.neuralnetworks.util.TensorFactory;
import com.github.neuralnetworks.util.UniqueList;

/**
 * Something similar to tensor factory
 */
public class ValuesProvider implements Serializable {

    private static final long serialVersionUID = 1L;

    private Set<Tensor> tensors;
    private Map<Object, List<Tensor>> values;

    public ValuesProvider(boolean useSharedMemory) {
	super();

	this.values = new HashMap<>();

	if (useSharedMemory) {
	    this.tensors = new HashSet<>();
	}
    }

    public ValuesProvider(Set<Tensor> tensors) {
	super();

	this.values = new HashMap<>();
	this.tensors = tensors;
    }

    /**
     * Get values for object based on provided dimensions
     * 
     * @param targetLayer
     * @param unitCount
     * @return
     */
    @SuppressWarnings("unchecked")
    public <T extends Tensor> T get(Object key, int... dimensions) {
	if (key != null && values.get(key) != null) {
	    if (dimensions == null || dimensions.length == 0) {
		if (values.get(key).size() == 1) {
		    return (T) values.get(key).iterator().next();
		} else if (values.get(key).size() > 1) {
		    throw new IllegalArgumentException("No dimensions provided");
		} else {
		    return null;
		}
	    }
	    
	    return (T) values.get(key).stream().filter(t -> Arrays.equals(t.getDimensions(), dimensions)).findFirst().orElse(null);
	}

	return null;
    }

    /**
     * Add tensor t with dimensions
     * @param key
     * @param dimensions
     */
    public void add(Object key, int... dimensions) {
	List<Tensor> set = values.get(key);
	if (set == null) {
	    values.put(key, set = new UniqueList<Tensor>());
	}

	if (useSharedMemory()) {
	    float[] elements = getElements();
	    if (elements == null) {
		elements = new float[0];
	    }

	    int l = elements.length;
	    float[] newElements = Arrays.copyOf(elements, l + Arrays.stream(dimensions).reduce(1, (a, b) -> a * b));
	    tensors.forEach(t -> t.setElements(newElements));
	    Tensor newTensor = TensorFactory.tensor(newElements, l, dimensions);
	    tensors.add(newTensor);
	    set.add(newTensor);
	} else {
	    set.add(TensorFactory.tensor(dimensions));
	}
    }


    /**
     * Add tensor t
     * @param key
     * @param t
     */
    public void add(Object key, Tensor t) {
	if (useSharedMemory() && t.getElements() != getElements()) {
	    throw new IllegalArgumentException("Tensor doesn't use the same base array");
	}

	List<Tensor> set = values.get(key);
	if (set == null) {
	    values.put(key, set = new UniqueList<Tensor>());
	}

	set.add(t);
    }

    public Set<Tensor> getTensors() {
        return tensors;
    }

    public boolean useSharedMemory() {
	return tensors != null;
    }

    private float[] getElements() {
	float[] elements = null;
	for (Tensor t : tensors) {
	    if (elements == null) {
		elements = t.getElements();
	    }

	    if (elements != t.getElements()) {
		elements = null;
		break;
	    }
	}

	return elements;
    }
}
