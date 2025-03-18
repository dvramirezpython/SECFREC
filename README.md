# SECFREC
Secure platform for fingertip matching and hand motion recognition on the fly

## Subproject Distal phalanx segmentation
> It segments distal phalanges from closer and farther distances. 
>
> It implements an AI-powered pipeline
> >
> > Starting with a fine-tuned YOLOv11s model for phalange segmentation,
> > Followed by a Pytorch implementation of FingerNet for fingerprint feature extraction.
> It repeats the fingerphoto capture until a fingerprint with sufficient feature existence score is acquired.
