package com.example.main;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.text.Font;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.io.*;
import java.sql.ResultSet;
import java.time.LocalDate;
import java.sql.Date;
import java.util.*;

public class Main extends Application
{
    //Stack to place all visited scenes on, so you can use the back button to go back
    private static Stack<Page> pages = new Stack<>();

    //TODO
    //Finalise neural net + process the information it gives into words
    //Can you remove need to pass scene? with stage.getScene(); ? can you push inside of change scene? would be spicy!!
    //TODO

    @Override
    public void start(Stage primaryStage) throws Exception
    {
        //Easy access to all fonts without having to constantly declare
        HashMap<String, Font> fonts = new HashMap<>();
        fonts.put("Title", new Font("Helvetica", 24));
        fonts.put("Subtitle", new Font("Helvetica", 16));
        fonts.put("Body", new Font ("Helvetica", 12));

        primaryStage.setTitle("Fingerspelling Search");
        primaryStage.setWidth(300);
        primaryStage.setHeight(360);

        //When window is closed
        primaryStage.setOnCloseRequest(e -> saveToSQL(primaryStage));

        Pane pane = new Pane();
        Scene scene = new Scene(pane);
        scene.setRoot(new StartPage(primaryStage, scene, fonts));
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public void saveToSQL(Stage stage)
    {
        //Get the user currently logged in
        Page page = (Page) stage.getScene().getRoot();
        UserProfile user = page.getUser();

        //Don't save anything if guest
        if(user.getUsername().equalsIgnoreCase("guest"))
        {
            return;
        }

        DatabaseConnection dbCon = new DatabaseConnection();
        dbCon.establishConnection();

        //If the user has no bookmarks then skip
        if(user.getBookmarks().size() != 0)
        {
            //Delete all current bookmarks from this user
            dbCon.runUpdate("delete from Bookmark where username='" + user.getUsername() + "'", -1);

            //Cycle through all the bookmarks and insert them into the SQL database
            String insertBookmarkCmd = "insert into Bookmark values";
            for (int i = 0; i < user.getBookmarks().size(); i++)
            {
                insertBookmarkCmd = insertBookmarkCmd + "('" + user.getUsername() + "','" + user.getBookmarks().get(i).getURL() + "','" + user.getBookmarks().get(i).getTitle() + "'),";
            }
            insertBookmarkCmd = insertBookmarkCmd.substring(0, insertBookmarkCmd.length() - 1);

            if(!dbCon.runUpdate(insertBookmarkCmd, user.getBookmarks().size()))
            {
                System.out.println("Error: Line 99");
            }
        }

        Queue<Website> history = user.getHistory();

        while(history.peek().getURL().equalsIgnoreCase("empty"))
        {
            //If the history was not filled all the way up, empty websites were added.
            //This removes them as we do not want to fill up the SQL database with blank records
            history.remove();
            if(history.size() == 0)
            {
                //If they have no history then leave
                dbCon.closeConnection();
                return;
            }
        }

        //Doing the same as with bookmarks, delete the current ones then add the new ones
        dbCon.runUpdate("delete from History where username='" + user.getUsername() + "'", -1);

        String insertHistoryCmd = "insert into History values";
        int position = 0;
        int size = history.size();
        for (int i = 0; i < size; i++)
        {
            Website tempHistory = history.remove();
            insertHistoryCmd = insertHistoryCmd + "('" + user.getUsername() + "','" + tempHistory.getURL() + "','" + tempHistory.getTitle() + "','" + Date.valueOf(tempHistory.getDateAccessed()) + "'," + (size - position) + "),";
            position++;
        }
        insertHistoryCmd = insertHistoryCmd.substring(0, insertHistoryCmd.length() - 1);

        if(!dbCon.runUpdate(insertHistoryCmd, position))
        {
            System.out.println("Error: Line 133");
        }

        dbCon.closeConnection();
    }

    public void changeScene(Stage stage, Scene scene, Page newPage)
    {
        newPage.setUser(pages.peek().getUser());
        newPage.createScene();
        scene.setRoot(newPage);
        stage.setWidth(newPage.getStageWidth());
        stage.setHeight(newPage.getStageHeight());
        stage.setScene(scene);
        stage.centerOnScreen();
    }

    class Page extends VBox
    {
        private int stageWidth;
        private int stageHeight;
        private UserProfile user;

        public int getStageWidth() { return stageWidth; }

        public void setStageWidth(int stageWidth) { this.stageWidth = stageWidth; }

        public int getStageHeight() { return stageHeight; }

        public void setStageHeight(int stageHeight) { this.stageHeight = stageHeight; }

        public UserProfile getUser() { return user; }

        public void setUser(UserProfile user) { this.user = user; }

        public void createScene(){}
    }

    //Place all pages on stack, for go back button
    class StartPage extends Page
    {
        StartPage(Stage stage, Scene scene, HashMap fonts)
        {

            setStageWidth(300);
            setStageHeight(360);
            setUser(new UserProfile());

            GridPane mainPane = new GridPane();
            mainPane.setVgap(10);
            mainPane.setHgap(10);

            Label welcomeToAppLbl = new Label("Fingerspelling Search");
            welcomeToAppLbl.setFont((Font) fonts.get("Title"));
            mainPane.add(welcomeToAppLbl,0 ,0);

            Label selectLogInOrSignUpLbl = new Label("Press below to log in/sign up!");
            selectLogInOrSignUpLbl.setFont((Font) fonts.get("Subtitle"));
            mainPane.add(selectLogInOrSignUpLbl, 0 ,1);

            Button logInBtn = new Button("Log In");
            logInBtn.setFont((Font) fonts.get("Body"));
            mainPane.add(logInBtn, 0, 2);
            logInBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new LogInPage(stage, scene, fonts));});

            Button signUpBtn = new Button("Sign Up");
            signUpBtn.setFont((Font) fonts.get("Body"));
            mainPane.add(signUpBtn, 0, 3);
            signUpBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new SignUpPage(stage, scene, fonts));});

            Label informationLbl = new Label("Fingerspelling Search is an A-Level Computer Science project, which aims to turn a video of BSL fingerspelling into text. It is very easy to get started, just create an account above and you will be able to see your past searches, as well as favourite any websites you frequent! If you want to know more about BSL, or just learn how to fingerspell, click the links below!");
            informationLbl.setFont((Font) fonts.get("Body"));
            informationLbl.maxWidth(300);
            informationLbl.setWrapText(true);
            mainPane.add(informationLbl, 0, 4);

            HBox links = new HBox(10);

            Button linkToBSLBtn = new Button("About BSL");
            linkToBSLBtn.setFont((Font) fonts.get("Body"));
            links.getChildren().add(linkToBSLBtn);
            linkToBSLBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new WebPage(stage, scene, fonts, "https://www.british-sign.co.uk/what-is-british-sign-language/"));});

            Button linkToFingerspellBtn = new Button("Fingerspelling Chart");
            linkToFingerspellBtn.setFont((Font) fonts.get("Body"));
            links.getChildren().add(linkToFingerspellBtn);
            linkToFingerspellBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new WebPage(stage, scene, fonts, "https://www.british-sign.co.uk/wp-content/uploads/2013/05/BSL-Fingerspelling-Right-Handed-1024x724.png"));});

            Button linkToFingerspellingGameBtn = new Button("Practice");
            linkToFingerspellingGameBtn.setFont((Font) fonts.get("Body"));
            links.getChildren().add(linkToFingerspellingGameBtn);
            linkToFingerspellingGameBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new WebPage(stage, scene, fonts, "https://www.british-sign.co.uk/fingerspelling-game/"));});

            this.getChildren().addAll(mainPane, links);
            this.setSpacing(5);
            this.setPadding(new Insets(2,0,0,10));
        }
    }

    class LogInPage extends Page
    {
        LogInPage(Stage stage, Scene scene, HashMap fonts)
        {
            setStageWidth(315);
            setStageHeight(190);

            GridPane logInPane = new GridPane();
            logInPane.setVgap(10);
            logInPane.setHgap(10);

            Label logInWelcomeLbl = new Label("Log In!");
            logInWelcomeLbl.setFont((Font) fonts.get("Title"));
            logInPane.add(logInWelcomeLbl,0,0);

            Label usernameLbl = new Label("Username:");
            usernameLbl.setFont((Font) fonts.get("Subtitle"));
            logInPane.add(usernameLbl, 0, 1);

            Label passwordLbl = new Label("Password:");
            passwordLbl.setFont((Font) fonts.get("Subtitle"));
            logInPane.add(passwordLbl,0,2);

            TextField usernameFld = new TextField();
            logInPane.add(usernameFld, 1, 1);

            PasswordField passwordFld = new PasswordField();
            logInPane.add(passwordFld,1, 2);

            Button logInBtn = new Button("Log in");
            logInBtn.setFont((Font) fonts.get("Body"));
            logInPane.add(logInBtn, 0, 3);
            logInBtn.setOnAction(e -> {
                if(checkLogIn(usernameFld.getText(), passwordFld.getText()))
                {
                    UserProfile user = new UserProfile(usernameFld.getText());

                    this.setUser(user);
                    usernameFld.clear();
                    passwordFld.clear();

                    loadHistory();
                    loadBookmarks();

                    Alert verifyCamera = new Alert(Alert.AlertType.CONFIRMATION, "In order to proceed, this app needs access to your camera, do you give permission?");
                    verifyCamera.setTitle("Camera Permissions");
                    verifyCamera.setHeaderText("Access Camera");

                    Optional<ButtonType> resultOfVerifyCamera = verifyCamera.showAndWait();
                    ButtonType buttonPressed = resultOfVerifyCamera.orElse(ButtonType.CANCEL);

                    if(buttonPressed == ButtonType.OK)
                    {
                        pages.push(this);
                        changeScene(stage, scene, new MainPage(stage, scene, fonts));
                    }
                }
            });

            Button signUpBtn = new Button("Sign up");
            signUpBtn.setFont((Font) fonts.get("Body"));
            logInPane.add(signUpBtn, 1, 3);
            signUpBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new SignUpPage(stage, scene, fonts));});

            Button backBtn = new Button("Back");
            logInBtn.setFont((Font) fonts.get("Body"));
            logInPane.add(backBtn, 2, 0);
            backBtn.setOnAction(e -> {changeScene(stage, scene, pages.peek()); pages.pop();});

            this.getChildren().addAll(logInPane);
            this.setPadding(new Insets(2,0,0,10));
        }

        private boolean checkLogIn(String username, String password)
        {

            if(username.equals("") || password.equals(""))
            {
                return false;
            }

            DatabaseConnection dbCon = new DatabaseConnection();
            dbCon.establishConnection();

            ResultSet validLogin = dbCon.runQuery("select * from RegisteredUser where username='" + username +"'");

            try
            {
                while(validLogin.next())
                {
                    String passwordTest = validLogin.getString("password");
                    if (passwordTest.equals(password))
                    {
                        dbCon.closeConnection();
                        return true;
                    }
                }
            }
            catch (Exception e)
            {
                System.out.println("Error: Line 195");
                e.printStackTrace();
                dbCon.closeConnection();
                return false;
            }

            Alert invalidLogin = new Alert(Alert.AlertType.INFORMATION, "You have not entered a valid username and password combination, please try again!");
            invalidLogin.setTitle("Incorrect Login");
            invalidLogin.setHeaderText("Incorrect Login");
            invalidLogin.showAndWait();

            dbCon.closeConnection();

            return false;
        }

        private void loadHistory()
        {
            DatabaseConnection dbCon = new DatabaseConnection();
            dbCon.establishConnection();

            ResultSet historyFromSQL = dbCon.runQuery("select * from History where username='" + this.getUser().getUsername() + "'");

            int size = 0;
            try
            {
                ResultSet countingSize = dbCon.runQuery("select count(*) from History where username='" + this.getUser().getUsername() + "'");
                if(countingSize.next())
                {
                    size = Integer.parseInt(countingSize.getString(1));
                }
            }
            catch (Exception e)
            {
                System.out.println("Error: line 269");
                e.printStackTrace();
            }


            Website[] websites = new Website[size];

            try
            {
                while(historyFromSQL.next())
                {
                    websites[historyFromSQL.getInt("position") - 1] = new Website(historyFromSQL.getString("URL"), historyFromSQL.getString("title"), historyFromSQL.getDate("dateAccessed").toLocalDate());
                }
            }
            catch (Exception e)
            {
                System.out.println("Error: Line 265");
                e.printStackTrace();
            }

            dbCon.closeConnection();

            Website placeholder = new Website("Empty", "Empty", LocalDate.of(1970, 1, 1));

            for (int i = size; i < 10; i++)
            {
                this.getUser().getHistory().add(placeholder);
            }

            for (int i = size - 1 ; i > -1; i--)
            {
                this.getUser().getHistory().add(websites[i]);
            }
        }

        private void loadBookmarks()
        {
            DatabaseConnection dbCon = new DatabaseConnection();
            dbCon.establishConnection();

            ResultSet bookmarksFromSQL = dbCon.runQuery("select * from Bookmark where username='" + this.getUser().getUsername() + "'");

            try
            {
                while(bookmarksFromSQL.next())
                {
                    this.getUser().getBookmarks().add(new Website(bookmarksFromSQL.getString("webLink"), bookmarksFromSQL.getString("title")));
                }
            }
            catch (Exception e)
            {
                System.out.println("Error: Line 291");
                e.printStackTrace();
            }

            dbCon.closeConnection();
        }
    }

    class SignUpPage extends Page
    {
        SignUpPage(Stage stage, Scene scene, HashMap fonts)
        {
            setStageWidth(330);
            setStageHeight(190);

            GridPane signUpPane = new GridPane();
            signUpPane.setVgap(10);
            signUpPane.setHgap(10);

            Label signUpWelcomeLbl = new Label("Sign Up!");
            signUpWelcomeLbl.setFont((Font) fonts.get("Title"));
            signUpPane.add(signUpWelcomeLbl,0,0);

            Label usernameLbl = new Label("Username:");
            usernameLbl.setFont((Font) fonts.get("Subtitle"));
            signUpPane.add(usernameLbl, 0, 1);

            Label passwordLbl = new Label("Password:");
            passwordLbl.setFont((Font) fonts.get("Subtitle"));
            signUpPane.add(passwordLbl,0,2);

            TextField usernameFld = new TextField();
            signUpPane.add(usernameFld, 1, 1);

            PasswordField passwordFld = new PasswordField();
            signUpPane.add(passwordFld,1, 2);

            Button logInBtn = new Button("Log in");
            logInBtn.setFont((Font) fonts.get("Body"));
            signUpPane.add(logInBtn, 1, 3);
            logInBtn.setOnAction(e -> {pages.push(this); changeScene(stage, scene, new LogInPage(stage, scene, fonts));});

            Button signUpBtn = new Button("Sign up");
            signUpBtn.setFont((Font) fonts.get("Body"));
            signUpPane.add(signUpBtn, 0, 3);
            signUpBtn.setOnAction(e -> {
                if(checkSignUp(usernameFld, passwordFld))
                {
                    UserProfile user = new UserProfile(usernameFld.getText());
                    pages.push(this);
                    changeScene(stage, scene, new LogInPage(stage, scene, fonts));
                }
            });

            Button backBtn = new Button("Back");
            backBtn.setFont((Font) fonts.get("Body"));
            signUpPane.add(backBtn, 2, 0);
            backBtn.setOnAction(e -> {changeScene(stage, scene, pages.peek()); pages.pop();});


            this.getChildren().addAll(signUpPane);
            this.setPadding(new Insets(2,0,0,10));
        }

        private Boolean checkSignUp(TextField usernameFld, PasswordField passwordFld)
        {
            String username = usernameFld.getText();
            String password = passwordFld.getText();

            usernameFld.clear();
            passwordFld.clear();

            if(username.equals("") || password.equals("") || username.equalsIgnoreCase("guest"))
            {
                Alert edgeCases = new Alert(Alert.AlertType.INFORMATION, "You have used a username or password that is not allowed, please choose again!");
                edgeCases.setTitle("Invalid Sign Up");
                edgeCases.setHeaderText("Invalid Details");
                edgeCases.showAndWait();
                return false;
            }

            DatabaseConnection dbCon = new DatabaseConnection();
            dbCon.establishConnection();

            ResultSet repeatedUsername = dbCon.runQuery("select * from RegisteredUser where username='" + username + "'");

            try
            {
                int counter = 0;
                while(repeatedUsername.next())
                {
                    counter++;
                }
                if(counter > 0)
                {
                    Alert cannotRepeat = new Alert(Alert.AlertType.INFORMATION, "You have the same username as another user! Please choose another");
                    cannotRepeat.setTitle("Invalid Sign Up");
                    cannotRepeat.setHeaderText("Invalid Username");
                    cannotRepeat.showAndWait();
                    dbCon.closeConnection();
                    return false;
                }
            }
            catch (Exception e)
            {
                System.out.println("Error: Line 298");
                e.printStackTrace();
                dbCon.closeConnection();
                return false;
            }

            if(dbCon.runUpdate("insert into RegisteredUser values('" + username + "', '" + password + "')", 1))
            {
                dbCon.closeConnection();
                return true;
            }

            dbCon.runUpdate("delete from RegisteredUser where username='" + username + "'", 1);

            Alert failedToAdd = new Alert(Alert.AlertType.ERROR, "The system was not able to sign you up! Please try again");
            failedToAdd.setTitle("Error 1");
            failedToAdd.setHeaderText("Registration Failure");
            failedToAdd.showAndWait();

            dbCon.closeConnection();

            return false;
        }
    }

    class MainPage extends Page
    {
        MainPage(Stage stage, Scene scene, HashMap fonts)
        {
            setStageWidth(527);
            setStageHeight(475);

            ImageView imageView = new ImageView();
            imageView.setFitHeight(384);
            imageView.setFitWidth(512);

            //grab the video of the default webcam
            VideoCapture webCam = new VideoCapture(1);

            //flag to show if camera is turned on or off, starts on as the user has agreed when logging in
            final Boolean[] hasStartedLiveFeed = {true};

            //whilst this animation is running, a new frame is constantly being captured from the webcam then displayed
            AnimationTimer animationTimerLiveFeed = new AnimationTimer()
            {
                @Override
                public void handle(long l)
                {
                    imageView.setImage(getCapture(webCam));
                }
            };
            animationTimerLiveFeed.start();

            Button cameraOnOffBtn = new Button("Camera On/Off");
            cameraOnOffBtn.setFont((Font) fonts.get("Body"));
            cameraOnOffBtn.setOnAction(e -> {
                //if camera off, turn it on and vice versa, remembering to change flags
                if(!hasStartedLiveFeed[0])
                {
                    hasStartedLiveFeed[0] = true;
                    animationTimerLiveFeed.start();
                    return;
                }
                hasStartedLiveFeed[0] = false;
                animationTimerLiveFeed.stop();
                imageView.setImage(new Image("black.jpg"));
            });

            //flag to denote a recording has started
            final Boolean[] hasStartedRecording = {false};
            //matrix that will hold all frames that are being recorded
            Mat videoRecordingFrame = new Mat();
            VideoWriter videoWriter = new VideoWriter();

            AnimationTimer animationTimerRecording = new AnimationTimer()
            {
                @Override
                public void handle(long l)
                {
                    //when recording, read and write frames to the file
                    webCam.read(videoRecordingFrame);
                    videoWriter.write(videoRecordingFrame);
                }
            };

            Button startRecordingBtn = new Button("Start Recording");
            startRecordingBtn.setFont((Font) fonts.get("Body"));
            startRecordingBtn.setOnAction(e -> {
                //if not recording, open a video file and start uploading the frames to it, and if recording, stop
                if(hasStartedLiveFeed[0])
                {
                    if(!hasStartedRecording[0])
                    {
                        startRecordingBtn.setText("Stop Recording");
                        videoWriter.open("video1.avi", VideoWriter.fourcc('M','J','P','G'), 15, new Size(webCam.get(Videoio.CAP_PROP_FRAME_WIDTH), webCam.get(Videoio.CAP_PROP_FRAME_HEIGHT)));
                        animationTimerRecording.start();
                        hasStartedRecording[0] = true;
                        return;
                    }

                    startRecordingBtn.setText("Start Recording");
                    videoWriter.release();
                    animationTimerRecording.stop();
                    hasStartedRecording[0] = false;

                    //get users text from the video
                    String userInput = processHands();

                    //asking if user wants to google search, save to text file or cancel
                    Alert searchOrSave = new Alert(Alert.AlertType.CONFIRMATION, "What would you like to do?");
                    searchOrSave.setTitle("Result Recorded");
                    searchOrSave.setHeaderText("Your text is: " + userInput.replace("+", " "));

                    ButtonType searchButton = new ButtonType("Search");
                    ButtonType saveButton = new ButtonType("Save");
                    ButtonType cancelButton = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);

                    searchOrSave.getButtonTypes().setAll(searchButton, saveButton, cancelButton);

                    Optional<ButtonType> result = searchOrSave.showAndWait();
                    if(result.get() == searchButton)
                    {
                        //if they want to search, create a new web page with their search as the site
                        changeScene(stage, scene, new WebPage(stage, scene, fonts, "https://www.google.com/search?q=" + userInput));
                        pages.push(this);
                    }
                    else if (result.get() == saveButton)
                    {
                        //show them their files, so they can choose where to save the file
                        DirectoryChooser directoryChooser = new DirectoryChooser();
                        directoryChooser.setInitialDirectory(new File("src"));
                        File selectedSaveLocation = directoryChooser.showDialog(stage);

                        try
                        {
                            //write to their chosen file location
                            BufferedWriter writer = new BufferedWriter(new FileWriter(selectedSaveLocation.getAbsolutePath() + "/text.txt"));
                            writer.write(userInput.replace("+", " "));
                            writer.close();
                        }
                        catch (Exception ex)
                        {
                            ex.printStackTrace();
                        }
                    }
                    else
                    {
                        //if they press cancel then close
                        searchOrSave.close();
                    }
                }
            });

            //when logging out, make sure to save things
            Button logOutBtn = new Button("Log Out");
            logOutBtn.setFont((Font) fonts.get("Body"));
            logOutBtn.setOnAction(e -> {saveToSQL(stage); changeScene(stage, scene, pages.pop()); animationTimerLiveFeed.stop(); animationTimerRecording.stop(); webCam.release();});

            //if they need help, display an information pop up
            Button helpBtn = new Button("Help");
            helpBtn.setFont((Font) fonts.get("Body"));
            helpBtn.setOnAction(e -> displayHelp());

            //button to go to history page
            Button historyBtn = new Button("History");
            historyBtn.setFont((Font) fonts.get("Body"));
            historyBtn.setOnAction(e -> {changeScene(stage, scene, new HistoryPage(stage, scene, fonts)); startRecordingBtn.setText("Start Recording"); videoWriter.release(); animationTimerRecording.stop(); hasStartedRecording[0] = false; pages.push(this);});

            //button to go to bookmarks page
            Button bookmarkBtn = new Button("Bookmarks");
            bookmarkBtn.setFont((Font) fonts.get("Body"));
            bookmarkBtn.setOnAction(e -> {changeScene(stage, scene, new BookmarksPage(stage, scene, fonts)); startRecordingBtn.setText("Start Recording"); videoWriter.release(); animationTimerRecording.stop(); hasStartedRecording[0] = false; pages.push(this);});

            HBox cameraAndRecording = new HBox(5);
            HBox historyAndBookmarks = new HBox(5);
            HBox helpAndLogOut = new HBox(5);

            cameraAndRecording.getChildren().addAll(cameraOnOffBtn, startRecordingBtn);
            historyAndBookmarks.getChildren().addAll(historyBtn, bookmarkBtn);
            helpAndLogOut.getChildren().addAll(helpBtn, logOutBtn);

            AnchorPane topRowPane = new AnchorPane();
            AnchorPane.setLeftAnchor(historyAndBookmarks, 5.0);
            AnchorPane.setRightAnchor(helpAndLogOut, 5.0);

            topRowPane.getChildren().addAll(historyAndBookmarks, helpAndLogOut);

            //adding all buttons to page
            this.getChildren().addAll(topRowPane, imageView, cameraAndRecording);
        }

        public void displayHelp()
        {
            Alert help = new Alert(Alert.AlertType.INFORMATION, "To record signing, press \"Start Recording\" and sign to the camera. When you finish a letter, put your left palm up, and when you finish a word put your right palm up. When you are finished with the recording, press \"Stop Recording\" and your signing will be processed!");
            help.setTitle("Help");
            help.setHeaderText("For Recording Sign");
            help.showAndWait();
        }

        public String processHands()
        {
            //declaring strings that will be edited in the loop
            String toReturn = "";
            String input = "";
            try
            {
                //run command in the command line of "py landmarks.py", which runs the script
                ProcessBuilder processBuilder = new ProcessBuilder("py", getFilePath("landmarks.py"));
                Process process = processBuilder.start();

                //read in the outputs from script we just ran
                BufferedReader result = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String landmarks = result.readLine();
                while(landmarks!=null)
                {
                    input = input + landmarks;
                    landmarks = result.readLine();
                }
                //remove the non-signs from the text
                input = input.replace(",", "");
                //split the input into words from where the user has signalled that there is a new word
                String[] words = input.split(":::::::::::::::");
                for (int i = 0; i < words.length; i++)
                {
                    //remove all extra new word symbols
                    words[i] = words[i].replace(":", "");
                    //split each word into letters from where the user has signalled that there is a new character
                    String[] letters = words[i].split(";;;;;;;;;;;;;;;");
                    for (int j = 0; j < letters.length; j++)
                    {
                        //remove all extra new character symbols
                        letters[j] = letters[j].replace(";", "");
                        //create an array corresponding to each letter
                        int[] countForEachLetter = new int[26];
                        for (int k = 0; k < letters[j].length(); k++)
                        {
                            //using the ascii value of each character, increment if there is an instance of a specific letter
                            countForEachLetter[(int) letters[j].charAt(k) - 97]++;
                        }
                        if(letters[j].length() != 0)
                        {
                            //add the most common character to the sentence to be returned
                            toReturn = toReturn + Character.toString(positionOfMaxValue(countForEachLetter) + 97);
                        }
                    }
                    //between words add a "+" as thats what happens in google's urls
                    toReturn = toReturn + "+";
                }
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
            return toReturn;
        }

        private int positionOfMaxValue(int[] arr)
        {
            int maxValuePos = 0;

            for (int i = 0; i < arr.length; i++)
            {
                if(arr[i] > arr[maxValuePos])
                {
                    maxValuePos = i;
                }
            }
            return maxValuePos;
        }

        public String getFilePath(String name)
        {
            File file = new File(name);
            return file.getAbsolutePath();
        }

        public Image getCapture(VideoCapture webCam)
        {
            //creating a new matrix to store pixel values in
            Mat mat = new Mat();
            webCam.read(mat);
            //after reading in, flipped so that the user can see themselves the 'right' way around
            Core.flip(mat, mat, 1);
            return convertingMatToImage(mat);
        }

        public Image convertingMatToImage(Mat mat)
        {
            MatOfByte byteMat = new MatOfByte();
            //convert the matrix of pixel numbers into a bitmap
            Imgcodecs.imencode(".bmp", mat, byteMat);
            //return the new image
            return new Image(new ByteArrayInputStream(byteMat.toArray()));
        }
    }

    class WebPage extends Page
    {
        private final Label userLogInStatusLbl;

        WebPage(Stage stage, Scene scene, HashMap fonts, String site)
        {
            setStageWidth(900);
            setStageHeight(600);

            HBox backForwardPane = new HBox(5);
            HBox historyBookmarksPane = new HBox(5);
            AnchorPane topBarPane = new AnchorPane();

            //Initialising the WebView and putting the search in
            WebView webView = new WebView();
            WebEngine engine = webView.getEngine();
            engine.load(site);

            //Contains the sites that were recently visited
            Stack<String> currentBackHistory = new Stack<>();
            //If you have used the back button, these are the sites that you pressed back on
            Stack<String> currentForwardHistory = new Stack<>();

            ImageView back = new ImageView(new Image("backArrow.png"));
            back.setFitHeight(20);
            back.setPreserveRatio(true);
            Button backToPreviousSiteBtn = new Button("", back);
            backToPreviousSiteBtn.setOnAction(e -> {
                //You pop off the stack twice, so you need the size to be at least 2
                if(currentBackHistory.size() > 1)
                {
                    //Push current site onto forward history stack
                    currentForwardHistory.push(currentBackHistory.pop());
                    //Load site you were on before
                    engine.load(currentBackHistory.pop());
                }
            });

            ImageView forward = new ImageView(new Image("forwardArrow.png"));
            forward.setFitHeight(20);
            forward.setPreserveRatio(true);
            Button forwardToPreviousSiteBtn = new Button("", forward);
            forwardToPreviousSiteBtn.setOnAction(e -> {
                if(!currentForwardHistory.empty())
                {
                    //Load the most recently put on site
                    engine.load(currentForwardHistory.pop());
                }
            });

            //Declaring the two different states that the save bookmark button can be in
            ImageView bookmarkNotFill = new ImageView(new Image("notBookmark.png"));
            bookmarkNotFill.setFitHeight(20);
            bookmarkNotFill.setPreserveRatio(true);
            ImageView bookmarkIsFill = new ImageView(new Image("isBookmark.png"));
            bookmarkIsFill.setFitHeight(20);
            bookmarkIsFill.setPreserveRatio(true);
            Button bookmarkBtn = new Button("", bookmarkNotFill);
            bookmarkBtn.setOnAction(e -> {
                //Checks that the user has an account, so is able to save things
                if(checkIfLoggedIn(stage, scene, fonts))
                {
                    //If this is a new bookmark, add it to the current list of bookmarks, remove it if it already is bookmarked!
                    if(bookmarkBtn.getGraphic() == bookmarkNotFill)
                    {
                        bookmarkBtn.setGraphic(bookmarkIsFill);
                        this.getUser().getBookmarks().add(new Website(engine.getLocation(), engine.getTitle()));
                        return;
                    }
                    bookmarkBtn.setGraphic(bookmarkNotFill);
                    this.getUser().removeBookmark(engine.getLocation(), engine.getTitle());
                }
            });

            //Listens for when a new site is loaded
            engine.getLoadWorker().stateProperty().addListener(e -> {
                //Adds to the history used by the back buttons
                currentBackHistory.push(engine.getLocation());
                //Adds to the user's history
                this.getUser().getHistory().add(new Website(engine.getLocation(), engine.getTitle(), java.time.LocalDate.now()));

                //If this site is currently bookmarked, it will set the save bookmark button to its 'saved' state, so that you cannot have the same bookmark twice
                if(this.getUser().containsBookmark(engine.getLocation(), engine.getTitle()))
                {
                    bookmarkBtn.setGraphic(bookmarkIsFill);
                    return;
                }
                //Resets the save bookmark button
                bookmarkBtn.setGraphic(bookmarkNotFill);

            });

            userLogInStatusLbl = new Label();
            userLogInStatusLbl.setFont((Font) fonts.get("Subtitle"));

            backForwardPane.getChildren().addAll(backToPreviousSiteBtn, forwardToPreviousSiteBtn, bookmarkBtn, userLogInStatusLbl);

            Button accessHistoryBtn = new Button("History");
            accessHistoryBtn.setFont((Font) fonts.get("Body"));
            accessHistoryBtn.setOnAction(e ->
            {
                if(checkIfLoggedIn(stage, scene, fonts))
                {
                    pages.push(this);
                    changeScene(stage, scene, new HistoryPage(stage, scene, fonts));
                }
            });

            Button accessBookmarksBtn = new Button("Bookmarks");
            accessBookmarksBtn.setFont((Font) fonts.get("Body"));
            accessBookmarksBtn.setOnAction(e ->
            {
                if(checkIfLoggedIn(stage, scene, fonts))
                {
                    pages.push(this);
                    changeScene(stage, scene, new BookmarksPage(stage, scene, fonts));
                }
            });

            Button exitBrowserBtn = new Button("Back");
            exitBrowserBtn.setFont((Font) fonts.get("Body"));
            exitBrowserBtn.setOnAction(e -> {changeScene(stage, scene, pages.peek()); pages.pop();});

            historyBookmarksPane.getChildren().addAll(accessHistoryBtn, accessBookmarksBtn, exitBrowserBtn);

            AnchorPane.setLeftAnchor(backForwardPane, 5.0);
            AnchorPane.setRightAnchor(historyBookmarksPane, 5.0);

            topBarPane.getChildren().addAll(backForwardPane, historyBookmarksPane);

            this.getChildren().addAll(topBarPane, webView);
        }

        public void createScene()
        {
            this.userLogInStatusLbl.setText("Logged in as: " + this.getUser().getUsername());
        }

        private boolean checkIfLoggedIn(Stage stage, Scene scene, HashMap fonts)
        {
            //If the user is logged in, return true
            if(this.getUser().getUsername().equalsIgnoreCase("guest"))
            {
                //Alert asking for them to log in
                Alert mustLogIn = new Alert(Alert.AlertType.CONFIRMATION, "You must be logged in to access this, would you like to log in?");
                mustLogIn.setTitle("Log In?");
                mustLogIn.setHeaderText("Signed in as Guest");

                Optional<ButtonType> resultOfMustLogIn = mustLogIn.showAndWait();
                ButtonType buttonPressed = resultOfMustLogIn.orElse(ButtonType.CANCEL);

                //If they press ok, they will be taken to the log in scene in order to log in
                if(buttonPressed == ButtonType.OK)
                {
                    pages.push(this);
                    changeScene(stage, scene, new LogInPage(stage, scene, fonts));
                }

                return false;
            }
            return true;
        }
    }

    class HyperlinkInCell extends Hyperlink
    {
        HyperlinkInCell(String title, String URL, Stage stage, Scene scene, HashMap fonts, Page page)
        {
            //Hyperlink text is the website title
            this.setText(title);
            //On click, create webpage and load site clicked
            this.setOnAction(e -> {
                pages.push(page); changeScene(stage, scene, new WebPage(stage, scene, fonts, URL));});
        }
    }

    class HistoryPage extends Page
    {
        TableView tableView = new TableView();

        Scene scene;
        Stage stage;
        HashMap fonts;
        HistoryPage(Stage stage, Scene scene, HashMap fonts)
        {
            setStageWidth(300);
            setStageHeight(350);

            this.scene = scene;
            this.stage = stage;
            this.fonts = fonts;

            Button backBtn = new Button("Back");
            backBtn.setFont((Font) fonts.get("Body"));
            backBtn.setOnAction(e -> {changeScene(stage, scene, pages.peek()); pages.pop();});

            tableView.prefHeightProperty().bind(stage.heightProperty());
            this.getChildren().addAll(backBtn,tableView);
        }

        public void createScene()
        {
            //Creating the columns to be put in the table, and creating hyperlinks
            TableColumn<Website, HyperlinkInCell> column1 = new TableColumn<>("Title");
            column1.setCellValueFactory(e -> new SimpleObjectProperty<>(new HyperlinkInCell(e.getValue().getTitle(), e.getValue().getURL(), stage, scene, fonts, this)));
            column1.setMinWidth(175);

            TableColumn<Website, String> column2 = new TableColumn<>("Date Visited");
            column2.setCellValueFactory(e -> new SimpleStringProperty(e.getValue().getDateAccessed().toString()));
            column2.setMinWidth(110);

            tableView.getColumns().addAll(column1, column2);
            //Adding the data needed for the table, reversed because history is stored back to front
            ObservableList<Website> list = FXCollections.observableArrayList(this.getUser().getHistory());
            FXCollections.reverse(list);
            tableView.setItems(list);
        }
    }

    class BookmarksPage extends Page
    {
        TableView tableView = new TableView();
        Stage stage;
        Scene scene;
        HashMap fonts;


        //THIS NEEDS TO BE BETTER, WORKS BUT NEEDS TO BE BETTER

        BookmarksPage(Stage stage, Scene scene, HashMap fonts)
        {
            setStageWidth(250);
            setStageHeight(400);

            this.stage = stage;
            this.scene = scene;
            this.fonts = fonts;

            Button backBtn = new Button("Back");
            backBtn.setFont((Font) fonts.get("Body"));
            backBtn.setOnAction(e -> {changeScene(stage, scene, pages.peek()); pages.pop();});

            tableView.prefHeightProperty().bind(stage.heightProperty());
            this.getChildren().addAll(backBtn, tableView);
        }

        public void createScene()
        {
            //Creating the column for the table, and adding hyperlinks
            TableColumn<Website, HyperlinkInCell> column1 = new TableColumn<>("Bookmark");
            column1.setCellValueFactory(e -> new SimpleObjectProperty<>(new HyperlinkInCell(e.getValue().getTitle(), e.getValue().getURL(), stage, scene, fonts, this)));
            column1.setMinWidth(235);

            tableView.getColumns().addAll(column1);
            //Giving the table the data
            tableView.setItems(FXCollections.observableArrayList(this.getUser().getBookmarks()));
        }
    }


    public static void main(String[] args) {
        nu.pattern.OpenCV.loadLocally();
        launch(args);
    }
}

