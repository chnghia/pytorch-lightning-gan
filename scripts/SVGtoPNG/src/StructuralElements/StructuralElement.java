package StructuralElements;

import java.awt.Polygon;
import java.util.ArrayList;

public class StructuralElement {

	private ArrayList<Polygon> polygons;
	private String name;
	
	public StructuralElement(String name)
	{
		this.name = name;
		polygons = new ArrayList<Polygon>();
	}
	
	public void addPolygon(Polygon p)
	{
		polygons.add(p);
	}
	
	public String getName()
	{
		return this.name;
	}
	
	public ArrayList<Polygon> getPolygons()
	{
		return this.polygons;
	}
}
