using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Practice_ML7_Node2.Server.Models;
using Tensorflow;
using Tensorflow.Clustering;
using static Tensorflow.Binding;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Math.Distances;


namespace Practice_ML7_Node2.Server.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ProductsController : ControllerBase
    {
        private readonly PrimaryDbContext _context;

        public ProductsController(PrimaryDbContext context)
        {
            _context = context;
        }

        // GET: api/Products
        [HttpGet]
        public async Task<ActionResult<IEnumerable<Product>>> GetProducts()
        {
            return await _context.Products.ToListAsync();
        }

        private readonly ILogger<ProductsController> _logger;
        public class PredictionData
        {
            public float PredictionDataUpdate { get; set; }
            public float AccordCentroid1 { get; set; }
            public float AccordCentroid2 { get; set; }
            public float AccordCentroid3 { get; set; }
        }

        /// <summary>
        /// An implementation of a simple API Method 
        /// Add specification for accepted parameters
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        [HttpGet("MLAction")]
        public async Task<ActionResult<Product>> MLAction(int id, string name)
        {
            try
            {
                /// <summary>
                /// factorypart1 is to get a record from the database
                /// Lets get a current object based upon the accepted
                /// parameter established in the method then pass the database
                /// object to each stage of the factory
                /// </summary>
                /// <param name=id></param>
                /// <param name=name></param>
                /// <returns></returns>

                /// 2) In this section we will obtain record aquired
                ///  based upon id passed as a param to the method
                var product = await _context.Products.FindAsync(id);

                if (product == null)
                {
                    return NotFound();
                }

                /// 4) Here we will establish the Factory and initalize 
                /// Above this method implement factorypart2 to design
                /// the factory
                /// Important: Implement factorypart2 above as new object
                var factory = ProductFactory.CreateFactory(id, name);
                
                /// 1) First we will define all the stages of the factory 
                /// and the params and database record based on id
                await factory.Stage1(id, name, _context);
                await factory.Stage2(id, name, _context);
                await factory.Stage3(id, name, _context);

                /// 3) Then we will return the object that is aquired based upon id
                /// 
               return Ok(product);
            }
            catch (Exception ex)

            {
                ///Return error message if the API method fails
                System.Diagnostics.Debug.WriteLine(ex.ToString()+": Error occurred during MLAction for product id: {ProductId}", id);
                return StatusCode(StatusCodes.Status500InternalServerError, "An error occurred while processing your request.");
            }
        }



        /// <summary>
        /// factorypart2 
        /// Establish the Factory Class: The locaction where all factory operations
        /// are conducted
        /// </summary>
        public class ProductFactory
        {
            /// <summary>
            /// Part  .5
            /// Explicity etablish the stages of the factory and like before
            /// define the parameters
            /// </summary>
            public interface IProductFactory
            {
                Task Stage1(int id, string name, PrimaryDbContext context);
                Task Stage2(int id, string name, PrimaryDbContext context);
                Task Stage3(int id, string name, PrimaryDbContext context);
            }

            /// <summary>
            /// Part  .7
            /// Explicity etablish the stages of the factory and like before
            /// define the parameters
            /// </summary>
            public static IProductFactory CreateFactory(int id, string name)
            {
                return new ConcreteProductFactory(id, name);
            }



            /// <summary>
            /// Part 1 
            /// Lets Create the actuall factory
            ///
            /// </summary>
            private class ConcreteProductFactory : IProductFactory
            {
                /// <summary>
                /// Part 2 
                /// Establish local Params to be used within
                /// ConcreteProductFactory
                /// </summary>
                private readonly int _id;
                private readonly string _name;
                private PredictionData _sharedPredictionData;

                /// <summary>
                /// Part 3 
                /// Establish an initalization of the class
                /// creating conventions for params 
                /// </summary>
                public ConcreteProductFactory(int id, string name)
                {
                    _id = id;
                    _name = name;

                }
                /// <summary>
                /// Part  4
                /// Define one of the stages of the factory 
                /// 
                /// </summary>
                public async Task Stage1(int id, string name, PrimaryDbContext context)
                {
                    /// <summary>
                    /// Part  10
                    /// Here we will implement the logic within the first stage of the factory 
                    /// Try: tflowpart1
                    /// </summary>
                    /// 

                    /// <summary>
                    ///  tflowpart1 
                    ///  Part 1
                    ///  Add the appropriate usings...
                    /// using Tensorflow;
                    /// using static Tensorflow.Binding;
                    /// </summary>
                    ///
                    System.Diagnostics.Debug.WriteLine($"Initializing Stage 1 method for id: {id}, name: {name}");
                    try
                    {
                        System.Diagnostics.Debug.WriteLine("Initializing TensorFlow operations");
                        /// <summary>
                        ///  tflowpart1 
                        ///  Part 2
                        ///  Enable TensorFlow eager execution to process graphs immediately
                        /// 
                        /// </summary>
                        tf.enable_eager_execution();
                        System.Diagnostics.Debug.WriteLine("TensorFlow eager execution enabled");


                        /// <summary>
                        ///  tflowpart1 
                        ///  Part 3
                        ///  Lets retrieve the approriate specified
                        /// model from the database
                        /// </summary>
                        System.Diagnostics.Debug.WriteLine("Fetching pricing model from database");
                        var pricingModel = await context.TrainingModels
                            ///Lets define the name of the model to find
                            .FirstOrDefaultAsync(m => m.ModelName == "Pricing_Model");

                        /// <summary>
                        ///  tflowpart1 
                        ///  Part 4
                        ///  We will Have a model store the prditiond
                        /// Lets add this object into the base model
                        /// and a logger that will log all the operations
                        /// 
                        /// private readonly ILogger<ProductsController> _logger;
                        /// public class PredictionData
                        ///     {
                        ///     public float PredictionDataUpdate { get; set; }
                        ///        }
                        /// 
                        /// </summary>
                        PredictionData predictionData = GetSharedPredictionData();

                        /// <summary>
                        ///  tflowpart1 
                        ///  Part 5
                        ///  We will evaluate if the ML model is found and create a condition
                        /// 
                        /// </summary>
                        if (pricingModel != null)
                        {
                            /// <summary>
                            ///  tflowpart1 
                            ///  Part 6
                            ///  In the circumstance where the model is found
                            ///  we will implememtn a procedure to train the model
                            ///  on the params that are passed 
                            /// Try: tflowpart1a
                            /// </summary>
                            /// 

                            /// <summary>
                            ///  tflowpart1a 
                            ///  Part 1
                            ///  Indicate the model is found and conduct training
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Existing pricing model found. Initializing fine-tuning process.");

                            /// <summary>
                            ///  tflowpart1a 
                            ///  Part 2
                            ///  Convert the model data into a memory stream
                            /// 
                            /// </summary>
                            using (var memoryStream = new MemoryStream(pricingModel.Data))
                            /// <summary>
                            ///  tflowpart1a 
                            ///  Part 3
                            ///  We will binary read the model as a memory stream
                            /// 
                            /// </summary>
                            using (var reader = new BinaryReader(memoryStream))
                            {
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 4
                                ///  Lets Deserialize the data
                                /// 
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Deserializing model parameters");
                                int wLength = reader.ReadInt32();
                                float[] wData = new float[wLength];
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 5
                                ///  Lets load the data into type variables for a TensorFlow
                                ///  model
                                /// </summary>
                                for (int i = 0; i < wLength; i++)
                                {
                                    wData[i] = reader.ReadSingle();
                                }
                                var W = tf.Variable(wData[0], dtype: TF_DataType.TF_FLOAT);

                                int bLength = reader.ReadInt32();
                                float[] bData = new float[bLength];
                                for (int i = 0; i < bLength; i++)
                                {
                                    bData[i] = reader.ReadSingle();
                                }
                                var b = tf.Variable(bData[0], dtype: TF_DataType.TF_FLOAT);
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 6
                                ///  After loading all the data into a new model
                                ///  indicate a sucessful load of model parameters
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Model parameters loaded successfully");
                                System.Diagnostics.Debug.WriteLine($"Initialized W value: {W.numpy()}, b value: {b.numpy()}");

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 7
                                ///  Lets use the data retrieved from the database as a constant
                                ///  to run aginst the model using TensorFlow operations
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Fetching product data for fine-tuning");
                                var productRecord = await context.Products
                                    ///Based upon the params we will find the record based on the provided id and name
                                    .Where(p => p.IdProduct == id && p.Name == name)
                                    .FirstOrDefaultAsync();

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 8
                                ///  If we look in the database and the product is not found then
                                ///  we will provide some error catching mechanism to prevent exceptions
                                ///  
                                /// </summary>
                                if (productRecord == null)
                                {
                                    System.Diagnostics.Debug.WriteLine($"Product initialization failed. Product with ID {id} and name '{name}' not found.");
                                    throw new ArgumentException($"Product with ID {id} and name '{name}' not found.");
                                }

                                System.Diagnostics.Debug.WriteLine($"Product data fetched. Price: {productRecord.Price}");

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 9
                                ///  Lets conduct TFlow operations to prepare the training data
                                ///  
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Initializing training data");
                                var trainData = tf.constant((float)productRecord.Price, dtype: TF_DataType.TF_FLOAT);
                                System.Diagnostics.Debug.WriteLine($"Training data initialized. Value: {trainData.numpy()}");

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 10
                                ///  Specify the level of tuning
                                ///  
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Initializing fine-tuning parameters");
                                int epochs = 50;
                                float learningRate = 1e-3f;

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 11
                                ///  Conduct machine learing operations to tune the model
                                ///  
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Starting fine-tuning process");
                                for (int epoch = 0; epoch < epochs; epoch++)
                                {
                                    try
                                    {
                                        using (var tape = tf.GradientTape())
                                        {
                                            ///Prepare loss procedure on training operations
                                            var predictions = tf.add(tf.multiply(trainData, W), b);
                                            var loss = tf.square(tf.subtract(predictions, trainData));

                                            var gradients = tape.gradient(loss, new[] { W, b });

                                            W.assign_sub(tf.multiply(gradients[0], tf.constant(learningRate)));
                                            b.assign_sub(tf.multiply(gradients[1], tf.constant(learningRate)));

                                            if (epoch % 10 == 0)
                                            {
                                                System.Diagnostics.Debug.WriteLine($"Fine-tuning Epoch {epoch}, Loss: {loss.numpy()}");
                                                System.Diagnostics.Debug.WriteLine($"Updated W value: {W.numpy()}, b value: {b.numpy()}");
                                            }
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        System.Diagnostics.Debug.WriteLine($"Error in fine-tuning loop at epoch {epoch}: {ex.Message}");
                                        System.Diagnostics.Debug.WriteLine($"Current W value: {W.numpy()}, b value: {b.numpy()}");
                                        System.Diagnostics.Debug.WriteLine($"Call Stack: {Environment.StackTrace}");
                                        throw new Exception($"Fine-tuning failed at epoch {epoch}", ex);
                                    }
                                }
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 12
                                ///  After training the model based on newly selected data calculate a prediction
                                ///  
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Fine-tuning completed. Calculating prediction.");
                                var prediction = tf.add(tf.multiply(trainData, W), b);
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 13
                                ///  Set the value of the PredictionDataUpdate
                                ///  object with the formulated prediction value
                                ///  
                                /// </summary>
                                predictionData.PredictionDataUpdate = prediction.numpy().ToArray<float>()[0];

                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 14
                                ///  After updating the PredictionDataUpdate
                                ///  object lets read the object value
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine($"Prediction calculated: {predictionData.PredictionDataUpdate}");


                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 15
                                ///  Lets prepare to save the model by applying serialization
                                ///  
                                ///  
                                /// </summary>
                                System.Diagnostics.Debug.WriteLine("Initializing model serialization");
                                var updatedModelData = new byte[0];
                                using (var saveStream = new MemoryStream())
                                {
                                    using (var writer = new BinaryWriter(saveStream))
                                    {
                                        writer.Write(1);
                                        writer.Write((float)W.numpy());
                                        writer.Write(1);
                                        writer.Write((float)b.numpy());
                                    }
                                    ///The model streamed to updatedModelData
                                    updatedModelData = saveStream.ToArray();
                                }

                                System.Diagnostics.Debug.WriteLine("Updating pricing model in database");
                                /// <summary>
                                ///  tflowpart1a 
                                ///  Part 16
                                ///  We will define the data in the catagory
                                ///  of "Data" in the TrainingModels table. 
                                ///  
                                /// </summary>
                                pricingModel.Data = updatedModelData;
                                ///Then save 
                                await context.SaveChangesAsync();
                                System.Diagnostics.Debug.WriteLine("Fine-tuned model saved to TrainingModels table");
                            }


                        }
                        else
                        {
                            /// <summary>
                            ///  tflowpart1 
                            ///  Part 7
                            ///  In the circumstance where the model is not found
                            ///  we will implememtn a procedure create a new model
                            ///  based upon what is in the database in terms of 
                            ///  the params that are passed
                            /// Try: tflowpart1b
                            /// </summary>
                            /// 

                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 1
                            ///  Indicate that there is no model found and 
                            ///  conduct the process to create the model and then train
                            ///  the model based  upon the data in the DB and the selection
                            ///  as a constant
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("No existing pricing model found. Initializing new model creation.");


                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 2
                            ///  Frist we will get the constant, this is aquaired from
                            ///  the databse context, then specified by the parameter that 
                            /// is established in the method
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Fetching product data");
                            var productRecord = await context.Products
                                .Where(p => p.IdProduct == id && p.Name == name)
                                .FirstOrDefaultAsync();


                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 3
                            ///  If after selecting the item from the database 
                            ///  for some reason the item cannot be found again generate an error message
                            /// 
                            /// </summary>
                            if (productRecord == null)
                            {
                                System.Diagnostics.Debug.WriteLine($"Product initialization failed. Product with ID {id} and name '{name}' not found.");
                                throw new ArgumentException($"Product with ID {id} and name '{name}' not found.");
                            }

                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 4
                            ///  Lets log the item selected by showing the item infomration  
                            ///  aquired from the database context
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine($"Product data fetched. Price: {productRecord.Price}");

                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 5
                            ///  Now we will take the param of the method and search by  
                            ///  a different criteria, in this case we will be searching by table
                            /// column, then selecting based upon the column specification
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Fetching all products with the same name for training");
                            var productsByName = await context.Products
                               .Where(p => p.Name == name)
                               .ToListAsync();
                            ///Lets load the results into a local variable 
                            var productsByNamePrices = productsByName.Select(p => (float)p.Price).ToArray();

                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 6
                            ///  From that list we will show the number of record that are aquired 
                            ///  Then we will clarify the range of of all the reconds aquired in terms
                            /// of a specified columns
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine($"Training data initialized. Number of samples: {productsByNamePrices.Length}");
                            System.Diagnostics.Debug.WriteLine($"Price range: {productsByNamePrices.Min()} to {productsByNamePrices.Max()}");


                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 8
                            ///  Lets prepare the training data  
                            ///  
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Initializing TensorFlow tensor");
                            Tensor trainData;
                            try
                            {
                                trainData = tf.convert_to_tensor(productsByNamePrices, dtype: TF_DataType.TF_FLOAT);
                                trainData = tf.reshape(trainData, new[] { -1, 1 }); // Reshape to 2D
                                System.Diagnostics.Debug.WriteLine($"Tensor shape initialized: {string.Join(", ", trainData.shape)}");
                            }
                            catch (Exception ex)
                            {
                                System.Diagnostics.Debug.WriteLine($"Tensor initialization failed: {ex.Message}");
                                throw new Exception("Failed to initialize tensor from price data.", ex);
                            }



                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 9 
                            ///  Lets prepare the training data  
                            ///  
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Initializing model variables");
                            var W = tf.Variable(tf.random.normal(new[] { 1, 1 }));
                            var b = tf.Variable(tf.zeros(new[] { 1 }));

                            System.Diagnostics.Debug.WriteLine($"Initial W shape: {string.Join(", ", W.shape)}, b shape: {string.Join(", ", b.shape)}");


                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 10 
                            ///  Then lets define the model inital specification  
                            ///  
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Initializing training parameters");
                            int epochs = 100;
                            float learningRate = 1e-2f;

                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 11 
                            ///  After etablising the constant, training data, and model design   
                            ///  lets conduct inital training process 
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Starting training process");
                            for (int epoch = 0; epoch < epochs; epoch++)
                            {
                                try
                                {
                                    using (var tape = tf.GradientTape())
                                    {
                                        var predictions = tf.matmul(trainData, W) + b;
                                        var loss = tf.reduce_mean(tf.square(predictions - trainData));

                                        var gradients = tape.gradient(loss, new[] { W, b });

                                        W.assign_sub(gradients[0] * learningRate);
                                        b.assign_sub(gradients[1] * learningRate);

                                        if (epoch % 10 == 0)
                                        {
                                            System.Diagnostics.Debug.WriteLine($"Training Epoch {epoch}, Loss: {loss.numpy()}");
                                        }
                                    }
                                }
                                catch (Exception ex)
                                {
                                    System.Diagnostics.Debug.WriteLine($"Error in training loop at epoch {epoch}: {ex.Message}");
                                    throw new Exception($"Training failed at epoch {epoch}", ex);
                                }
                            }
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 12 
                            ///  After the model is created from data from the database then trained    
                            ///  lets prepare the prediction data  
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Training completed. Preparing for prediction.");
                            var inputArray = new float[] { (float)productRecord.Price };
                            var inputTensor = tf.convert_to_tensor(inputArray, dtype: TF_DataType.TF_FLOAT);
                            inputTensor = tf.reshape(inputTensor, new[] { -1, 1 }); // Reshape to 2D
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 13 
                            ///  After the prediction is created we will update the PredictionDataUpdate
                            ///  object with the predicted value
                            ///   
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Calculating prediction");
                            var prediction = tf.matmul(inputTensor, W) + b;
                            predictionData.PredictionDataUpdate = prediction.numpy().ToArray<float>()[0];
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 14 
                            ///  First lets reference the original retrieved record
                            ///  Then we will reference the object field that has just been udated 
                            ///   
                            /// 
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine($"Prediction calculated. Original price: {productRecord.Price}, Predicted price: {predictionData.PredictionDataUpdate}");
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 15 
                            ///  Then we will serialize the trained model to save to the database
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Initializing model serialization");
                            var modelData = new byte[0];
                            using (var memoryStream = new MemoryStream())
                            {
                                using (var writer = new BinaryWriter(memoryStream))
                                {
                                    var wData = W.numpy().ToArray<float>();
                                    writer.Write(wData.Length);
                                    foreach (var value in wData)
                                    {
                                        writer.Write(value);
                                    }

                                    var bData = b.numpy().ToArray<float>();
                                    writer.Write(bData.Length);
                                    foreach (var value in bData)
                                    {
                                        writer.Write(value);
                                    }
                                }
                                ///Then we will store the serialized model into a variable
                                modelData = memoryStream.ToArray();
                            }
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 16 
                            ///  Lets approprate columns of the new record to be created
                            /// </summary>
                            System.Diagnostics.Debug.WriteLine("Creating new TrainingModel entry");
                            var trainingModel = new TrainingModel
                            {
                                ModelName = "Pricing_Model",
                                Data = modelData
                            };
                            /// <summary>
                            ///  tflowpart1b 
                            ///  Part 17 
                            ///  Here we will save the trained model to the database
                            /// </summary>
                            context.TrainingModels.Add(trainingModel);
                            await context.SaveChangesAsync();

                            System.Diagnostics.Debug.WriteLine("New model saved to TrainingModels table");

                        }

                        System.Diagnostics.Debug.WriteLine($"Final PredictionDataUpdate value: {predictionData.PredictionDataUpdate}");
                        System.Diagnostics.Debug.WriteLine("Stage 1 method completed successfully");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Stage 1 method failed: {ex.Message}");
                        System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
                        throw;
                    }

                    await Task.Run(() => System.Diagnostics.Debug.WriteLine($"Stage1: {id}, {name}"));
                }
                /// <summary>
                /// Part  5
                /// Define one of the stages of the factory 
                /// 
                /// </summary>
                public async Task Stage2(int id, string name, PrimaryDbContext context)
                {
                    System.Diagnostics.Debug.WriteLine("Starting Stage2 method");

                    // Retrieve the PredictionData instance from a shared location
                    // This could be a property of the factory class or passed as a parameter
                    PredictionData predictionData = GetSharedPredictionData();

                    // Log the PredictionDataUpdate value from Stage 1
                    System.Diagnostics.Debug.WriteLine($"PredictionDataUpdate from Stage 1: {predictionData.PredictionDataUpdate}");

                    // Fetch products
                    System.Diagnostics.Debug.WriteLine("Fetching products from database");
                    var productsByName = await context.Products
                       .Where(p => p.Name == name)
                       .ToListAsync();

                    System.Diagnostics.Debug.WriteLine($"Found {productsByName.Count} products with name '{name}'");
                    foreach (var product in productsByName)
                    {
                        System.Diagnostics.Debug.WriteLine($"Name: {product.Name}, Price: {product.Price:F4}");
                    }

                    // Extract prices and convert to double
                    System.Diagnostics.Debug.WriteLine("Extracting prices for clustering");
                    var prices = productsByName.Select(p => new double[] { (double)p.Price }).ToArray();

                    // Define clustering parameters
                    int numClusters = 3; // Ensure we always have 3 clusters
                    int numIterations = 100;
                    System.Diagnostics.Debug.WriteLine($"Clustering parameters: clusters={numClusters}, iterations={numIterations}");

                    // Create k-means algorithm
                    var kmeans = new Accord.MachineLearning.KMeans(numClusters)
                    {
                        MaxIterations = numIterations,
                        Distance = new SquareEuclidean()
                    };

                    // Compute the algorithm
                    System.Diagnostics.Debug.WriteLine("Starting k-means clustering");
                    var clusters = kmeans.Learn(prices);

                    // Get the cluster centroids
                    var centroids = clusters.Centroids;

                    System.Diagnostics.Debug.WriteLine("K-means clustering completed");

                    // Get cluster assignments for each point
                    var assignments = clusters.Decide(prices);

                    // Log final results
                    System.Diagnostics.Debug.WriteLine("Final clustering results:");
                    for (int i = 0; i < prices.Length; i++)
                    {
                        System.Diagnostics.Debug.WriteLine($"Price: {prices[i][0]:F4}, Cluster: {assignments[i]}");
                    }

                    System.Diagnostics.Debug.WriteLine("Final centroids:");
                    for (int i = 0; i < numClusters; i++)
                    {
                        System.Diagnostics.Debug.WriteLine($"Centroid {i}: {centroids[i][0]:F4}");
                    }

                    // Set centroid values to PredictionData object
                    predictionData.AccordCentroid1 = (float)centroids[0][0];
                    predictionData.AccordCentroid2 = (float)centroids[1][0];
                    predictionData.AccordCentroid3 = (float)centroids[2][0];

                    // Calculate a simple metric: average distance to assigned centroid
                    double totalDistance = 0;
                    for (int i = 0; i < prices.Length; i++)
                    {
                        double distance = Math.Abs(prices[i][0] - centroids[assignments[i]][0]);
                        totalDistance += distance;
                    }
                    double avgDistance = totalDistance / prices.Length;

                    System.Diagnostics.Debug.WriteLine($"Average distance to assigned centroid: {avgDistance:F4}");

                    // Log all PredictionData values
                    System.Diagnostics.Debug.WriteLine("PredictionData values:");
                    System.Diagnostics.Debug.WriteLine($"PredictionDataUpdate: {predictionData.PredictionDataUpdate}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid1: {predictionData.AccordCentroid1}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid2: {predictionData.AccordCentroid2}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid3: {predictionData.AccordCentroid3}");

                    System.Diagnostics.Debug.WriteLine("Stage2 method completed");
                }

                // This method should be added to the ConcreteProductFactory class
                private PredictionData GetSharedPredictionData()
                {
                    // If this doesn't exist yet, create it
                    if (_sharedPredictionData == null)
                    {
                        _sharedPredictionData = new PredictionData();
                    }
                    return _sharedPredictionData;
                }

                /// <summary>
                /// Part  7
                /// Define one of the stages of the factory 
                /// 
                /// </summary>
                public async Task Stage3(int id, string name, PrimaryDbContext context)
                {
                    System.Diagnostics.Debug.WriteLine("Starting Stage3 method");

                    // Retrieve the shared PredictionData instance
                    PredictionData predictionData = GetSharedPredictionData();

                    // Output the requested values
                    System.Diagnostics.Debug.WriteLine("PredictionData values in Stage 3:");
                    System.Diagnostics.Debug.WriteLine($"PredictionDataUpdate: {predictionData.PredictionDataUpdate}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid1: {predictionData.AccordCentroid1}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid2: {predictionData.AccordCentroid2}");
                    System.Diagnostics.Debug.WriteLine($"AccordCentroid3: {predictionData.AccordCentroid3}");

                    // Perform any Stage 3 specific operations here
                    // For now, we'll just output the id and name
                    await Task.Run(() => System.Diagnostics.Debug.WriteLine($"Stage3 processing for id: {id}, name: {name}"));

                    // You can add more Stage 3 specific logic here

                    System.Diagnostics.Debug.WriteLine("Stage3 method completed");
                }
                /// <summary>
                /// Part  9
                /// After the stages are complete we will put together a product based
                /// upon what we get back from the stages
                /// </summary>
                // Keeping this method as it might be used elsewhere

            }///End ConcreteProductFactory : IProductFactory


        }///End ProductFactory

    }
}
