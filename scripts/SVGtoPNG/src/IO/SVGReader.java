package IO;

import java.awt.Polygon;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import StructuralElements.StructuralElement;


public class SVGReader {

	private ArrayList<StructuralElement> elements;
	private StructuralElement backgroundElement;
	private ArrayList<String> elementsName;
	private File fileInput;

	// Class constructor
	public SVGReader(File inFile)
	{
		fileInput = inFile;
	}

	// Reads the GT and 
	public void readSVG() throws IOException
	{
		FileReader finput = new FileReader(this.fileInput);
		BufferedReader svg = new BufferedReader(finput);
		String line, eName;
		elementsName = new ArrayList<String>();
		elements = new ArrayList<StructuralElement>();
		backgroundElement = new StructuralElement("Background");

		// read the first line
		line = svg.readLine();

		while(!line.contains("<class>"))
			line = svg.readLine();

		// read the different element names in the SVG file
		while(line.contains("<class>"))
		{
			eName = line.substring(7, line.length()-8);
			if(eName.equals("Wall") || eName.equals("Room") || eName.equals("Window") ||
					eName.equals("Separation") || eName.equals("Door") || eName.equals("Parking"))
			{
				elementsName.add(eName);
				elements.add(new StructuralElement(eName));
			}
			line = svg.readLine();
		}

		// while is not the end of the file
		while(!line.contains("</svg>") && !line.contains("<relation"))
		{
			// read the type of element
			int index1 = line.indexOf('"');
			int index2 = line.indexOf('"',index1+1);
			
			ArrayList<Integer> x = new ArrayList<Integer>();
			ArrayList<Integer> y = new ArrayList<Integer>();
			// get the position in the array of this element;
			String name = line.substring(index1+1,index2);
			int position = elementsName.indexOf(name);
			if (position != -1)
			{
				// get the beginning of the line points
				int linePosition = line.indexOf("points=");
				// create the polygon
				index1 = linePosition+8;
				String subs;
				// while do not arrive to the end of the line, read the numbers
				while(true)
				{
					index2 = line.indexOf(",",index1);
					if(index2==-1)
						break;
					subs = line.substring(index1, index2);
					double num = Double.parseDouble(subs);
					x.add((int) Math.round(num));
					index1 = index2+1;
					index2 = line.indexOf(" ",index1);
					subs = line.substring(index1, index2);
					num = Double.parseDouble(subs);
					y.add((int) Math.round(num));
					index1 = index2+1;
				}
				// create the arrays for extracting the polygon points
				int[] xP = new int[x.size()];
				int[] yP = new int[y.size()];
				for(int n = 0; n<xP.length; n++)
				{
					xP[n] = x.get(n);
					yP[n] = y.get(n);
				}

				// create the polygon and added to the specified StructuralElement
				Polygon p = new Polygon(xP, yP, xP.length);
				elements.get(position).addPolygon(p);
				// add the polygon to the background element
				backgroundElement.addPolygon(p);
			}
			line = svg.readLine();
		}
	}


	// This functions returns the elements read in svg file
	public ArrayList<StructuralElement> getElements()
	{
		return this.elements;
	}

	// This functions returns the elements read in svg file
	public StructuralElement getBackgroundElement()
	{
		return this.backgroundElement;
	}

}
