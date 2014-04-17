package com.github.neuralnetworks.util;

import java.util.HashMap;
import java.util.Map;

/**
 * Properties class for convenience
 */
public class Properties extends HashMap<String, Object> {

    private static final long serialVersionUID = 1L;

    public Properties() {
	super();
    }

    public Properties(int initialCapacity, float loadFactor) {
	super(initialCapacity, loadFactor);
    }

    public Properties(int initialCapacity) {
	super(initialCapacity);
    }

    public Properties(Map<? extends String, ? extends Object> m) {
	super(m);
    }

    @SuppressWarnings("unchecked")
    public <T> T getParameter(String key) {
	Object o = get(key);
	return o != null ? (T) o : null;
    }

    public <T> void setParameter(String name, T value) {
	if (value != null) {
	    put(name, value);
	} else {
	    remove(name);
	}
    }
}
