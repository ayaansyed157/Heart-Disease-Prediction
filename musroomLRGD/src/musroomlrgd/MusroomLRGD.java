
package musroomlrgd;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;
import java.util.Scanner;


public class MusroomLRGD extends JPanel{

    private static final double LEARNING_RATE = .01;
    private static final int ITERATIONS = 1000;

    private final List<double[]> features = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();
    
    private double[] weights;
    private double bias;

    public void readData(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                double[] featureMatrix = new double[parts.length - 1];
                boolean hasNonNumeric = false;
                for (int i = 0; i < featureMatrix.length; i++) {
                    try {
                        featureMatrix[i] = Double.parseDouble(parts[i]);
                    } catch (NumberFormatException e) {
                        hasNonNumeric = true;
                        break;
                    }
                }
                if (!hasNonNumeric) {
                    features.add(featureMatrix);
                    labels.add(Integer.valueOf(parts[parts.length - 1]));
                }
            }       
        }
    }

   public void train() {
        int numFeatures = features.get(0).length;
        //System.out.print(numFeatures);
        weights = new double[numFeatures];
        for(int i =0;i<weights.length;i++){
            
       weights[i]=Math.random() * 0.01 - 0.005;
   }
        bias = Math.random() * 0.01 - 0.005;
        
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            double[] gradient = new double[numFeatures];
            
            for (int i = 0; i < features.size(); i++) {
                
                double[] featureMatrix = features.get(i);
                int label = labels.get(i);
                
                double prediction = predict(featureMatrix);
                //System.out.println(prediction);
                
                for (int j = 0; j < numFeatures; j++) {
                    gradient[j] += (label - prediction) * featureMatrix[j];
                    //System.out.println(gradient[j]);
                }
            }
            for (int j = 0; j < numFeatures; j++) {
                weights[j] += LEARNING_RATE * gradient[j];
               //System.out.println(weights[j]);
            }
        }
        
       
    }

    public double predict(double[] featureMatrix) {
        double z = bias;
        for (int i = 0; i < featureMatrix.length; i++) {
            //dot product of wTx
            
            z += weights[i] * featureMatrix[i];
            
            }
            
        return sigmoid(z);
    }

    private double sigmoid(double z) {
        
        return 1.0 / (1.0 + Math.exp(-z));
    }
  
    public double calculateAccuracy(double[][] testFeatures, int[] testLabels) 
    {
       int correctPredictions = 0;
       for (int i = 0; i < testFeatures.length; i++) {
           double[] featureMatrix = testFeatures[i];
           int predictedLabel = predict(featureMatrix) >= 0.5 ? 1 : 0;
           if (predictedLabel == testLabels[i]) {
               correctPredictions++;
           }
       }
       //System.out.println(testFeatures.length);
       return (double) correctPredictions / testFeatures.length;
    }

    public static void main(String[] args) {
        MusroomLRGD lr = new MusroomLRGD();
        try {
            lr.readData("framingham.csv"); // Replace with your dataset file
            lr.train();
            //lr.plotData();
            
            // Testing accuracy
            List<double[]> testFeatures = new ArrayList<>();
            List<Integer> testLabels = new ArrayList<>();
                        
            for (int i = 0; i < lr.features.size(); i++) {
                double[] featureMatrix = lr.features.get(i);
                int label = lr.labels.get(i);
                testFeatures.add(featureMatrix);
                testLabels.add(label);
            }
            
            double[][] testFeaturesArray = testFeatures.toArray(new double[0][0]);
            int[] testLabelsArray = testLabels.stream().mapToInt(Integer::intValue).toArray();
        
            double accuracy = lr.calculateAccuracy(testFeaturesArray, testLabelsArray);
            System.out.println("Accuracy: " + accuracy);
            
            //Predicting based off of user input
            Scanner scanner = new Scanner(System.in);
            System.out.print("Enter age: ");
            double age = scanner.nextDouble();
            System.out.print("Enter glucose level: ");
            double glucose = scanner.nextDouble();
            System.out.print("Enter BMI level: ");
            double BMI = scanner.nextDouble();
            double[] testFeature = {age, glucose, BMI};
            double risk = lr.predict(testFeature);
            //double risk2 = Math.random();
            
            if(risk>0.5){
            System.out.println("Likely to be at risk of heart attack: " + risk);
            }
            else{
            System.out.println("Likely to be not at risk of heart attack: " + risk);
            }
           //System.out.println("Actual "+risk);
            
            
            } catch (IOException e) {
                e.printStackTrace();
            }
    

        
        
    }




}
    


