/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.object_track_detection.detection;
//WE use mConnectedThread to write to bluetooth

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.object_track_detection.detection.customview.OverlayView;
import org.tensorflow.lite.object_track_detection.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.object_track_detection.detection.env.BorderedText;
import org.tensorflow.lite.object_track_detection.detection.env.ImageUtils;
import org.tensorflow.lite.object_track_detection.detection.env.Logger;
import org.tensorflow.lite.object_track_detection.detection.tflite.Classifier;
import org.tensorflow.lite.object_track_detection.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.object_track_detection.detection.tracking.MultiBoxTracker;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Method;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.UUID;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  //BLUEOOTH STUFF
  private TextView mBluetoothStatus;
  private TextView mReadBuffer;
  private Button mScanBtn;
  private Button mOffBtn;
  private Button mListPairedDevicesBtn;
  private Button mDiscoverBtn;
  private BluetoothAdapter mBTAdapter;
  private Set<BluetoothDevice> mPairedDevices;
  private ArrayAdapter<String> mBTArrayAdapter;
  private ListView mDevicesListView;
  private CheckBox mLED1;

  private Handler mHandler; // Our main handler that will receive callback notifications
  private ConnectedThread mConnectedThread; // bluetooth background worker thread to send and receive data
  private BluetoothSocket mBTSocket = null; // bi-directional client-to-client data path

  private static final UUID BTMODULEUUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"); // "random" unique identifier


  // #defines for identifying shared types between calling functions
  private final static int REQUEST_ENABLE_BT = 1; // used to identify adding bluetooth names
  private final static int MESSAGE_READ = 2; // used in bluetooth handler to identify message update
  private final static int CONNECTING_STATUS = 3; // used in bluetooth handler to identify message status


  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    mBluetoothStatus = (TextView)findViewById(R.id.bluetoothStatus);
    mReadBuffer = (TextView) findViewById(R.id.readBuffer);
    mScanBtn = (Button)findViewById(R.id.scan);
    mOffBtn = (Button)findViewById(R.id.off);
    mDiscoverBtn = (Button)findViewById(R.id.discover);
    mListPairedDevicesBtn = (Button)findViewById(R.id.PairedBtn);
    mLED1 = (CheckBox)findViewById(R.id.checkboxLED1);

    mBTArrayAdapter = new ArrayAdapter<String>(this,android.R.layout.simple_list_item_1);
    mBTAdapter = BluetoothAdapter.getDefaultAdapter(); // get a handle on the bluetooth radio

    mDevicesListView = (ListView)findViewById(R.id.devicesListView);
    mDevicesListView.setAdapter(mBTArrayAdapter); // assign model to view
    mDevicesListView.setOnItemClickListener(mDeviceClickListener);


    // Ask for location permission if not already allowed
    if(ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED)
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.ACCESS_COARSE_LOCATION}, 1);


    mHandler = new Handler(){
      public void handleMessage(android.os.Message msg){
        if(msg.what == MESSAGE_READ){
          String readMessage = null;
          try {
            readMessage = new String((byte[]) msg.obj, "UTF-8");
          } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
          }
          mReadBuffer.setText(readMessage);
        }

        if(msg.what == CONNECTING_STATUS){
          if(msg.arg1 == 1)
            mBluetoothStatus.setText("Connected to Device: " + (String)(msg.obj));
          else
            mBluetoothStatus.setText("Connection Failed");
        }
      }
    };

    if (mBTArrayAdapter == null) {
      // Device does not support Bluetooth
      mBluetoothStatus.setText("Status: Bluetooth not found");
      Toast.makeText(getApplicationContext(),"Bluetooth device not found!",Toast.LENGTH_SHORT).show();
    }
    else {
//THIS IS WHERE WE RIGHT (LED TEST)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
      mLED1.setOnClickListener(new View.OnClickListener(){
        @Override
        public void onClick(View v){
          if(mConnectedThread != null) //First check to make sure thread created
            mConnectedThread.write("1");
        }
      });


      mScanBtn.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          bluetoothOn(v);
        }
      });

      mOffBtn.setOnClickListener(new View.OnClickListener(){
        @Override
        public void onClick(View v){
          bluetoothOff(v);
        }
      });

      mListPairedDevicesBtn.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v){
          listPairedDevices(v);
        }
      });

      mDiscoverBtn.setOnClickListener(new View.OnClickListener(){
        @Override
        public void onClick(View v){
          discover(v);
        }
      });
    }
  }
  private void bluetoothOn(View view){
    if (!mBTAdapter.isEnabled()) {
      Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
      startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT);
      mBluetoothStatus.setText("Bluetooth enabled");
      Toast.makeText(getApplicationContext(),"Bluetooth turned on",Toast.LENGTH_SHORT).show();

    }
    else{
      Toast.makeText(getApplicationContext(),"Bluetooth is already on", Toast.LENGTH_SHORT).show();
    }
  }

  // Enter here after user selects "yes" or "no" to enabling radio
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent Data){
    // Check which request we're responding to
    if (requestCode == REQUEST_ENABLE_BT) {
      // Make sure the request was successful
      if (resultCode == RESULT_OK) {
        // The user picked a contact.
        // The Intent's data Uri identifies which contact was selected.
        mBluetoothStatus.setText("Enabled");
      }
      else
        mBluetoothStatus.setText("Disabled");
    }
  }

  private void bluetoothOff(View view){
    mBTAdapter.disable(); // turn off
    mBluetoothStatus.setText("Bluetooth disabled");
    Toast.makeText(getApplicationContext(),"Bluetooth turned Off", Toast.LENGTH_SHORT).show();
  }

  private void discover(View view){
    // Check if the device is already discovering
    if(mBTAdapter.isDiscovering()){
      mBTAdapter.cancelDiscovery();
      Toast.makeText(getApplicationContext(),"Discovery stopped",Toast.LENGTH_SHORT).show();
    }
    else{
      if(mBTAdapter.isEnabled()) {
        mBTArrayAdapter.clear(); // clear items
        mBTAdapter.startDiscovery();
        Toast.makeText(getApplicationContext(), "Discovery started", Toast.LENGTH_SHORT).show();
        registerReceiver(blReceiver, new IntentFilter(BluetoothDevice.ACTION_FOUND));
      }
      else{
        Toast.makeText(getApplicationContext(), "Bluetooth not on", Toast.LENGTH_SHORT).show();
      }
    }
  }

  final BroadcastReceiver blReceiver = new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
      String action = intent.getAction();
      if(BluetoothDevice.ACTION_FOUND.equals(action)){
        BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
        // add the name to the list
        mBTArrayAdapter.add(device.getName() + "\n" + device.getAddress());
        mBTArrayAdapter.notifyDataSetChanged();
      }
    }
  };

  private void listPairedDevices(View view){
    mBTArrayAdapter.clear();
    mPairedDevices = mBTAdapter.getBondedDevices();
    if(mBTAdapter.isEnabled()) {
      // put it's one to the adapter
      for (BluetoothDevice device : mPairedDevices)
        mBTArrayAdapter.add(device.getName() + "\n" + device.getAddress());

      Toast.makeText(getApplicationContext(), "Show Paired Devices", Toast.LENGTH_SHORT).show();
    }
    else
      Toast.makeText(getApplicationContext(), "Bluetooth not on", Toast.LENGTH_SHORT).show();
  }

  private AdapterView.OnItemClickListener mDeviceClickListener = new AdapterView.OnItemClickListener() {
    public void onItemClick(AdapterView<?> av, View v, int arg2, long arg3) {

      if(!mBTAdapter.isEnabled()) {
        Toast.makeText(getBaseContext(), "Bluetooth not on", Toast.LENGTH_SHORT).show();
        return;
      }

      mBluetoothStatus.setText("Connecting...");
      // Get the device MAC address, which is the last 17 chars in the View
      String info = ((TextView) v).getText().toString();
      final String address = info.substring(info.length() - 17);
      final String name = info.substring(0,info.length() - 17);

      // Spawn a new thread to avoid blocking the GUI one
      new Thread()
      {
        public void run() {
          boolean fail = false;

          BluetoothDevice device = mBTAdapter.getRemoteDevice(address);

          try {
            mBTSocket = createBluetoothSocket(device);
          } catch (IOException e) {
            fail = true;
            Toast.makeText(getBaseContext(), "Socket creation failed", Toast.LENGTH_SHORT).show();
          }
          // Establish the Bluetooth socket connection.
          try {
            mBTSocket.connect();
          } catch (IOException e) {
            try {
              fail = true;
              mBTSocket.close();
              mHandler.obtainMessage(CONNECTING_STATUS, -1, -1)
                      .sendToTarget();
            } catch (IOException e2) {
              //insert code to deal with this
              Toast.makeText(getBaseContext(), "Socket creation failed", Toast.LENGTH_SHORT).show();
            }
          }
          if(fail == false) {
            mConnectedThread = new ConnectedThread(mBTSocket);
            mConnectedThread.start();

            mHandler.obtainMessage(CONNECTING_STATUS, 1, -1, name)
                    .sendToTarget();
          }
        }
      }.start();
    }
  };

  private BluetoothSocket createBluetoothSocket(BluetoothDevice device) throws IOException {
    try {
      final Method m = device.getClass().getMethod("createInsecureRfcommSocketToServiceRecord", UUID.class);
      return (BluetoothSocket) m.invoke(device, BTMODULEUUID);
    } catch (Exception e) {
      //Log.e(TAG, "Could not create Insecure RFComm Connection",e);

    }
    return  device.createRfcommSocketToServiceRecord(BTMODULEUUID);
  }

  private class ConnectedThread extends Thread {
    private final BluetoothSocket mmSocket;
    private final InputStream mmInStream;
    private final OutputStream mmOutStream;

    public ConnectedThread(BluetoothSocket socket) {
      mmSocket = socket;
      InputStream tmpIn = null;
      OutputStream tmpOut = null;

      // Get the input and output streams, using temp objects because
      // member streams are final
      try {
        tmpIn = socket.getInputStream();
        tmpOut = socket.getOutputStream();
      } catch (IOException e) { }

      mmInStream = tmpIn;
      mmOutStream = tmpOut;
    }

    public void run() {
      byte[] buffer = new byte[1024];  // buffer store for the stream
      int bytes; // bytes returned from read()
      // Keep listening to the InputStream until an exception occurs
      while (true) {
        try {
          // Read from the InputStream
          bytes = mmInStream.available();
          if(bytes != 0) {
            buffer = new byte[1024];
            SystemClock.sleep(100); //pause and wait for rest of data. Adjust this depending on your sending speed.
            bytes = mmInStream.available(); // how many bytes are ready to be read?
            bytes = mmInStream.read(buffer, 0, bytes); // record how many bytes we actually read
            mHandler.obtainMessage(MESSAGE_READ, bytes, -1, buffer)
                    .sendToTarget(); // Send the obtained bytes to the UI activity
          }
        } catch (IOException e) {
          e.printStackTrace();

          break;
        }
      }
    }

    /* Call this from the main activity to send data to the remote device */
    public void write(String input) {
      byte[] bytes = input.getBytes();           //converts entered String into bytes
      try {
        mmOutStream.write(bytes);
      } catch (IOException e) { }
    }

    /* Call this from the main activity to shutdown the connection */
    public void cancel() {
      try {
        mmSocket.close();
      } catch (IOException e) { }
    }
  }
  private static String getPath(String file, Context context) {
    AssetManager assetManager = context.getAssets();
    BufferedInputStream inputStream = null;
    try {
      // Read data from assets.
      inputStream = new BufferedInputStream(assetManager.open(file));
      byte[] data = new byte[inputStream.available()];
      inputStream.read(data);
      inputStream.close();
      // Create copy file in storage.
      File outFile = new File(context.getFilesDir(), file);
      FileOutputStream os = new FileOutputStream(outFile);
      os.write(data);
      os.close();
      // Return a path to file which may be read in common way.
      return outFile.getAbsolutePath();
    } catch (IOException ex) {
      Log.e("Mistake: ", "Failed to upload a file");
    }
    return "";
  }
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                LOGGER.i("Running detection on image " + currTimestamp);
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas = new Canvas(cropCopyBitmap);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(2.0f);

                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                switch (MODE) {
                  case TF_OD_API:
                    minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                    break;
                }

                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();

                for (final Classifier.Recognition result : results) {
                  final RectF location = result.getLocation();
                  if (location != null && result.getConfidence() >= minimumConfidence) {
                    canvas.drawRect(location, paint);
                    //>>>>>>First Trial Edit<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    if (result.getTitle().equals("cup")) //THIS WORKS!!!!
                    {
                      cropToFrameTransform.mapRect(location);

                      result.setLocation(location);
                      mappedRecognitions.add(result);
                    }
                  }
                }

                tracker.trackResults(mappedRecognitions, currTimestamp);
                trackingOverlay.postInvalidate();

                computingDetection = false;

                runOnUiThread(
                        new Runnable() {
                          @Override
                          public void run() {
                            showFrameInfo(previewWidth + "x" + previewHeight);
                            showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                            showInference(lastProcessingTimeMs + "ms");
                          }
                        });
              }
            });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}


