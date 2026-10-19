
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

    // Must match the feature columns in framingham.csv, before TenYearCHD.
    private static final String[] FEATURE_NAMES = {
        "male", "age", "education", "currentSmoker", "cigsPerDay",
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    };
    private static final String[] FEATURE_PROMPTS = {
        "male (0 or 1)", "age (years)", "education (1-4)",
        "current smoker (0 or 1)", "cigarettes per day",
        "blood pressure medication (0 or 1)", "history of stroke (0 or 1)",
        "hypertension (0 or 1)", "diabetes (0 or 1)",
        "total cholesterol (mg/dL)", "systolic blood pressure (mmHg)",
        "diastolic blood pressure (mmHg)", "BMI (kg/m^2)",
        "heart rate (beats/min)", "glucose (mg/dL)"
    };

    private final List<double[]> features = new ArrayList<>();
    private final List<Integer> labels = new ArrayList<>();
    
    private double[] weights;
    private double bias;
    private double[] featureMeans;
    private double[] featureScales;

    private void fitFeatureScaler() {
        int numFeatures = features.get(0).length;
        featureMeans = new double[numFeatures];
        featureScales = new double[numFeatures];
        for (double[] row : features) {
            for (int j = 0; j < numFeatures; j++) {
                featureMeans[j] += row[j];
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            featureMeans[j] /= features.size();
        }
        for (double[] row : features) {
            for (int j = 0; j < numFeatures; j++) {
                double difference = row[j] - featureMeans[j];
                featureScales[j] += difference * difference;
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            double standardDeviation = Math.sqrt(featureScales[j] / features.size());
            // A constant column becomes zero after centering; avoid division by zero.
            featureScales[j] = standardDeviation == 0.0 ? 1.0 : standardDeviation;
        }
    }

    private double scaleFeature(double value, int column) {
        return (value - featureMeans[column]) / featureScales[column];
    }

    public void readData(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String expectedHeader = String.join(",", FEATURE_NAMES) + ",TenYearCHD";
            if (!expectedHeader.equals(reader.readLine())) {
                throw new IOException("Dataset columns must match: " + expectedHeader);
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",", -1);
                if (parts.length != FEATURE_NAMES.length + 1) {
                    throw new IOException("Expected 15 features and one label per row.");
                }
                double[] featureMatrix = new double[FEATURE_NAMES.length];
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
        if (features.isEmpty()) {
            throw new IllegalStateException("Load at least one valid training row first.");
        }
        // Fit only on training records. Keep the original values for prediction input.
        fitFeatureScaler();
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
                    gradient[j] += (label - prediction) * scaleFeature(featureMatrix[j], j);
                    //System.out.println(gradient[j]);
                }
            }
            for (int j = 0; j < numFeatures; j++) {
                weights[j] += LEARNING_RATE * gradient[j]/features.size();
               //System.out.println(weights[j]);
            }
        }
        
       
    }

    public double predict(double[] featureMatrix) {
        if (weights == null) {
            throw new IllegalStateException("Train the model before predicting.");
        }
        if (featureMatrix == null || featureMatrix.length != weights.length) {
            throw new IllegalArgumentException("Expected " + weights.length
                    + " features in dataset column order.");
        }
        double z = bias;
        for (int i = 0; i < featureMatrix.length; i++) {
            //dot product of wTx
            
            z += weights[i] * scaleFeature(featureMatrix[i], i);
            
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
            double[] testFeature = new double[FEATURE_NAMES.length];
            for (int i = 0; i < FEATURE_NAMES.length; i++) {
                System.out.print("Enter " + FEATURE_PROMPTS[i] + ": ");
                testFeature[i] = scanner.nextDouble();
            }
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
    


