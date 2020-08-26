package program;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import IO.SVGReader;
import StructuralElements.StructuralElement;
import StructuralElements.myImage;

public class SVG2PNG {


	private static String inSVGFolder;
	private static String inOriginalFolder;
	private static String outGT;

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {

		File currentDir = new File("..");
		inOriginalFolder = currentDir.getCanonicalPath() + "\\originalImages";
		inSVGFolder = currentDir.getCanonicalPath() + "\\svgImages";
		outGT = currentDir.getCanonicalPath() + "\\GT";
		
		// create the outFolder if it does not exist
		File directory = new File(outGT);
		directory.mkdir();

		// read the original files
		File originalFiles = new File(inOriginalFolder);
		File[] originalListOfFiles = originalFiles.listFiles();

		// read the original files
		File SVGFiles = new File(inSVGFolder);
		File[] SVGListOfFiles = SVGFiles.listFiles();


		// for every original file
		for (int i = 0; i < originalListOfFiles.length; i++)
		{
			// get the name of the image
			String actualImageName = originalListOfFiles[i].getName();
			String actualImageNameWithoutExtension = actualImageName.substring(0, actualImageName.length()-4);
			actualImageNameWithoutExtension = actualImageNameWithoutExtension.concat("_gt");

			// read the original image
			File actualImageFile = originalListOfFiles[i];
			BufferedImage img = ImageIO.read(actualImageFile);

			// get the width an the height of the image
			double imageWidth = img.getWidth();
			double imageHeight = img.getHeight();

			// SVG variables
			File actualSVGFile = null;
			String SVGImageNameWithotExtension = null;
			String SVGImageName = null;
			int j = 0;
			
			// find the corresponding SVG image for the original image
			while(!actualImageNameWithoutExtension.equals(SVGImageNameWithotExtension))
			{
				SVGImageName = SVGListOfFiles[j].getName();
				SVGImageNameWithotExtension = SVGImageName.substring(0, actualImageNameWithoutExtension.length());
				j++;
			}
			// read the SVG file
			actualSVGFile = SVGListOfFiles[j-1];
			SVGReader svg = new SVGReader(actualSVGFile);
			svg.readSVG();
			// get the elements found
			ArrayList<StructuralElement> elements = svg.getElements();
			// recompose the original name of the image
			SVGImageNameWithotExtension = SVGImageNameWithotExtension.substring(0, SVGImageNameWithotExtension.length()-3); 
			// for every structural elements store the image
			for(int n = 0; n < elements.size(); n++)
			{
				myImage image = new myImage((int)imageHeight,(int)imageWidth);
				// for every structural elements store the image
				// get the structural element
				StructuralElement element = elements.get(n);
				// draw the polygons
				image.drawPolygons(element.getPolygons());
				// get the elements name
				String elementName = element.getName();
				// create the new file
				File finalImageFile = new File(outGT + "\\" + SVGImageNameWithotExtension + "-" + elementName + ".png"); 
				// store the image
				image.writeImage(finalImageFile, "png");
			}
			myImage image = new myImage((int)imageHeight,(int)imageWidth);
			// do the sam for the background element
			StructuralElement backgroundElement = svg.getBackgroundElement();
			// draw the polygons in the opposite color
			image.drawPolygons(backgroundElement.getPolygons(),1);
			// get the elements name
			String elementName = backgroundElement.getName();
			// create the new file
			File finalImageFile = new File(outGT + "\\" + SVGImageNameWithotExtension + "-" + elementName + ".png"); 
			// store the image
			image.writeImage(finalImageFile, "png");

		}
		System.out.println("Succeful PNG GT Generation");
	}

}
