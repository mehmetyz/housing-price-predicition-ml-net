using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace CaliforniaHousingPricePredictions
{
    public class HousingData
    {
        [LoadColumn(0)] public float Longitude { get; set; }
        [LoadColumn(1)] public float Latitude { get; set; }
        [LoadColumn(2)] public float HousingMedianAge { get; set; }
        [LoadColumn(3)] public float TotalRooms { get; set; }
        [LoadColumn(4)] public float TotalBedrooms { get; set; }
        [LoadColumn(5)] public float Population { get; set; }
        [LoadColumn(6)] public float Households { get; set; }
        [LoadColumn(7)] public float MedianIncome { get; set; }
        [LoadColumn(8)] public float MedianHouseValue { get; set; }
        [LoadColumn(9)] public string OceanProximity { get; set; }
    }

    class Program
    {
        private static string GetWorkingDirectory()
        {
            string path = Environment.CurrentDirectory;
            for (int i = 0; i < 3; i++)
            {
                path = Path.Combine(path, "..");
            }

            return path;
        }

        private static readonly string dataPath = Path.Combine(GetWorkingDirectory(), "housing.csv");

        static void Main(string[] args)
        {
            MLContext context = new MLContext();

            IDataView housingData = context.Data
                .LoadFromTextFile<HousingData>(dataPath, hasHeader: true, separatorChar: ',');

            TrainTestData splitDataView = context.Data.TrainTestSplit(housingData, testFraction: 0.2);

            IDataView trainData = splitDataView.TrainSet;
            IDataView testData = splitDataView.TestSet;

            var encodeOceanProximityTransform = context.Transforms.Categorical.OneHotEncoding("OceanProximity");
            var featureTransform = context.Transforms.Concatenate("Features", "Longitude", "Latitude",
                "HousingMedianAge", "TotalRooms", "TotalBedrooms", "Population", "Households", "MedianIncome");

            var dataProcessingPipeline = encodeOceanProximityTransform
                .Append(featureTransform)
                .AppendCacheCheckpoint(context);

            // Create and train model
            var trainer = context.Regression.Trainers.Sdca(labelColumnName: "MedianHouseValue", featureColumnName: "Features");
            var trainingPipeline = dataProcessingPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainData);

            // Evaluate model
            var predictions = trainedModel.Transform(testData);
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: "MedianHouseValue", scoreColumnName: "Score");

            Console.WriteLine($"\x1b[92m R Squared: {metrics.RSquared}");
            Console.WriteLine($"\x1b[91m RootMeanSquaredError: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"\x1b[95m MeanSquaredError: {metrics.MeanSquaredError}");
            Console.WriteLine($"\x1b[39m");

        }
    }
}
