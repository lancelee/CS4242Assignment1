package assignment1;

import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;

/*
 * This simple example shows you how to read the data in json format and get the specific field of a data sample.
 * Here, we extract the textual content of a tweet, split a sentence to words, and then convert the words to integer index.
 * For example, the following sentence "AA BB AA CC DD BB" will be converted to "0 1 0 2 3 1"
 * 
 * In this example, the words are split only with blank space. Therefore, they may not be actual words: they may contain
 * punctuation, mentions(@someone), URL, etc. You may need to improve this function. Besides, other techniques to process
 * the texts, e.g., removing stop words and low frequency words, could also be applied.
 * 
 * Enjoy your first Assignment :)
 * 
 * TA: Bian Jingwen (bjwpku@gmail.com)
 * 
 */

public class GetWordIndex {
	
	public static void main(String args[]){
		
		// The path of the input file where the original tweets are stored in json format
		String inputFileName = "TRAIN/NUS1_TRAIN.txt";
		
		// The path of the output file to save the result
		String outputFileName = "outputdata/NUS1_textualWords.txt";

		try{
			// Reader to read the input file
			BufferedReader reader = new BufferedReader(new FileReader(inputFileName));			
			// Writer to write the result file
			BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));
		    
			// This HashMap save the mapping relationship between a textual word and its index number
			HashMap<String, Integer> wordToIndex = new HashMap<String, Integer>();
			
			String line;
			while((line = reader.readLine()) != null){
				
				// Convert the input String line to JSONObject
				JSONObject tweet = new JSONObject(line);
				
				// Get the "text" field, which saves the textual content of this tweet
				// Other fields could be fetched similarly
				String text = tweet.getString("text");
				
				// Split this sentence into words
				// Here, the sentence is split only with blank space
				// You may consider other factors, e.g., punctuation, word root, URL, mentions(@someone)...
				String[] words = text.toLowerCase().split("\\s+");
				
				for(int i=0; i<words.length; i++){
					
					// Map a textual word to an integer index number
					if(!wordToIndex.containsKey(words[i])){
						wordToIndex.put(words[i], wordToIndex.size());
					}
					
					// Output the mapped words to file
					writer.write(wordToIndex.get(words[i])+" ");
				}
				
				writer.write("\n");
			}
			
			reader.close();
			writer.close();
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
}
