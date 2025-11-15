import React, { useState, useEffect, useCallback } from "react";
import {
  makeStyles,
  withStyles,
} from "@material-ui/core/styles";
import {
  AppBar,
  Toolbar,
  Typography,
  Avatar,
  Container,
  Card,
  CardContent,
  Paper,
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
} from "@material-ui/core";
import { common } from "@material-ui/core/colors";
import Clear from "@material-ui/icons/Clear";
import { DropzoneArea } from "material-ui-dropzone";
import axios from "axios";
import potatologo from "./potatologo.png";
import bgImage from "./bg.png";

// ----------------------------------------------------
// 🎨 Custom Button
// ----------------------------------------------------
const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    "&:hover": {
      backgroundColor: "#ffffff7a",
    },
  },
}))(Button);

// ----------------------------------------------------
// 🧩 Styles
// ----------------------------------------------------
const useStyles = makeStyles((theme) => ({
  grow: { flexGrow: 1 },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  media: { height: 400 },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  mainContainer: {
    backgroundImage: `url(${bgImage})`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundSize: "cover",
    height: "93vh",
    marginTop: "8px",
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    height: 500,
    backgroundColor: "transparent",
    boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 30%) !important",
    borderRadius: "15px",
  },
  appbar: {
    background: "#be6a77",
    boxShadow: "none",
    color: "white",
  },
  loader: {
    color: "#be6a77 !important",
  },
  tableCell: {
    fontSize: "22px",
    backgroundColor: "transparent !important",
    borderColor: "transparent !important",
    color: "#000000a6 !important",
    fontWeight: "bolder",
    padding: "1px 24px 1px 16px",
  },
  tableCell1: {
    fontSize: "14px",
    backgroundColor: "transparent !important",
    borderColor: "transparent !important",
    color: "#000000a6 !important",
    fontWeight: "bolder",
    padding: "1px 24px 1px 16px",
  },
  detail: {
    backgroundColor: "white",
    display: "flex",
    justifyContent: "center",
    flexDirection: "column",
    alignItems: "center",
  },
}));

// ----------------------------------------------------
// 🚀 Main Component
// ----------------------------------------------------
export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [hasImage, setHasImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // ✅ API Base URL (use .env or fallback)
  const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

  // ✅ Send File to FastAPI backend
  const sendFile = useCallback(async () => {
    if (hasImage && selectedFile) {
      try {
        setIsLoading(true);
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await axios.post(`${API_BASE}/predict`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        if (res.status === 200) {
          setData(res.data);
        }
      } catch (err) {
        console.error("❌ Error uploading file:", err);
        alert("Prediction failed! Please check backend connection.");
      } finally {
        setIsLoading(false);
      }
    }
  }, [hasImage, selectedFile, API_BASE]); // dependencies

  // ✅ Clear all data
  const clearData = () => {
    setData(null);
    setHasImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  // ✅ Generate preview for uploaded image
  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  // ✅ Trigger prediction automatically once preview is ready
  useEffect(() => {
    if (preview) sendFile();
  }, [preview, sendFile]);

  // ✅ Handle file selection from dropzone
  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setHasImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setHasImage(true);
  };

  const confidence = data
    ? (parseFloat(data.confidence) * 100).toFixed(2)
    : 0;

  return (
    <>
      {/* Navbar */}
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography variant="h6" noWrap>
            Potato Disease Classifier
          </Typography>
          <div className={classes.grow} />
          <Avatar src={potatologo} />
        </Toolbar>
      </AppBar>

      {/* Main Container */}
      <Container
        maxWidth={false}
        className={classes.mainContainer}
        disableGutters={true}
      >
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={classes.imageCard}>
              {/* Image Preview */}
              {hasImage && (
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={preview}
                    component="img"
                    title="Potato Leaf"
                  />
                </CardActionArea>
              )}

              {/* Upload Area */}
              {!hasImage && (
                <CardContent>
                  <DropzoneArea
                    acceptedFiles={["image/*"]}
                    dropzoneText={
                      "Drag and drop an image of a potato leaf to classify"
                    }
                    onChange={onSelectFile}
                  />
                </CardContent>
              )}

              {/* Prediction Result */}
              {data && (
                <CardContent className={classes.detail}>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell className={classes.tableCell1}>
                            Label:
                          </TableCell>
                          <TableCell
                            align="right"
                            className={classes.tableCell1}
                          >
                            Confidence:
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell
                            component="th"
                            scope="row"
                            className={classes.tableCell}
                          >
                            {data.class}
                          </TableCell>
                          <TableCell
                            align="right"
                            className={classes.tableCell}
                          >
                            {confidence}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              )}

              {/* Loader */}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress color="secondary" />
                  <Typography variant="h6" noWrap>
                    Processing...
                  </Typography>
                </CardContent>
              )}
            </Card>
          </Grid>

          {/* Clear Button */}
          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton
                variant="contained"
                className={classes.clearButton}
                onClick={clearData}
                startIcon={<Clear fontSize="large" />}
              >
                Clear
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </>
  );
};
