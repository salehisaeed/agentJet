/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2016-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "agentJetFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "probes.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::agentJetFvPatchVectorField::initializeFaceMapping()
{
    faceActionMapping_.setSize(patch().size(), 0);
    jetDirection_.setSize(patch().size());
    const vectorField& patchNormal = patch().nf();

    if (dict_.found("faceActionMapping") && dict_.found("jetDirections"))
    {
        faceActionMapping_ = dict_.lookup("faceActionMapping");
        jetDirection_ = vectorField("jetDirections", dict_, patch().size());
        return;
    }

    // Handle single-action case
    if (nActions_ == 1)
    {
        if (dict_.found("jetDirection"))
        {
            jetDirection_ = vectorField("jetDirection", dict_, patch().size());
            forAll(jetDirection_, i)
            {
                scalar magDir = mag(jetDirection_[i]);
                if (magDir > SMALL)
                {
                    jetDirection_[i] /= magDir;
                }
                else
                {
                    FatalErrorInFunction << "Injection jetDirection magnitude is too small"
                                         << abort(FatalError);
                }
            }
        }
        else
        {
            jetDirection_ = patchNormal;
            Info << "No jetDirection specified. Using patch normal direction." << endl;
        }
        return;
    }

    // Multi-action case (nActions_ > 1) requires multiActionMapping dictionary
    if (!dict_.found("multiActionMapping"))
    {
        FatalErrorInFunction
            << "'multiActionMapping' must be provided when nActions > 1."
            << abort(FatalError);
    }

    const List<dictionary> mappingEntries = dict_.lookup("multiActionMapping");

    forAll(mappingEntries, regionI)
    {
        const dictionary& regionDict = mappingEntries[regionI];

        vector minCorner, maxCorner, jetDir;
        regionDict.lookup("minCorner") >> minCorner;
        regionDict.lookup("maxCorner") >> maxCorner;
        label actionIdx = regionDict.lookupOrDefault<label>("actionIndex", -1);

        bool hasJetDirection = regionDict.found("jetDirection");
        if (hasJetDirection)
        {
            regionDict.lookup("jetDirection") >> jetDir;
            scalar magDir = mag(jetDir);
            if (magDir > SMALL)
            {
                jetDir /= magDir;
            }
            else
            {
                FatalErrorInFunction << "jetDirection magnitude too small for region " << regionI
                                    << abort(FatalError);
            }
        }
        else
        {
            Info << "No jetDirection specified for region " << regionI
                 << ". Using patch normal direction instead." << endl;
        }

        forAll(patch(), faceI)
        {
            const vector faceCenter = patch().Cf()[faceI];

            if
            (
                faceCenter.x() >= minCorner.x() && faceCenter.x() <= maxCorner.x() &&
                faceCenter.y() >= minCorner.y() && faceCenter.y() <= maxCorner.y() &&
                faceCenter.z() >= minCorner.z() && faceCenter.z() <= maxCorner.z()
            )
            {
                faceActionMapping_[faceI] = actionIdx;
                jetDirection_[faceI] = hasJetDirection ? jetDir : patchNormal[faceI];
            }
        }
    }
}


Foam::scalarField Foam::agentJetFvPatchVectorField::environmentState()
{
    const fvMesh& mesh = patch().boundaryMesh().mesh();

    dictionary probesDict;
    probesDict.add("type", probes::typeName);
    probesDict.add("fixedLocations", "true");
    probesDict.add("fields", "()"); // The field should defined in the BC dictionary
    probesDict.add("interpolationScheme", interpolationScheme_);
    probesDict.add("probeLocations", stateProbeLocations_);
    probes p
    (
        "probes",
        internalField().mesh().time(),
        probesDict
    );

    Foam::Field<scalar> sampledState;

    if (mesh.foundObject<volScalarField>(stateFieldName_))
    {
        sampledState = p.sample<scalar>(stateFieldName_);
    }
    else if (mesh.foundObject<volVectorField>(stateFieldName_))
    {
        Foam::vectorField sampledVector = p.sample<vector>(stateFieldName_);

        // Flatten vector field into scalar field (component-wise)
        sampledState.setSize(sampledVector.size() * vector::nComponents);
        forAll(sampledVector, i)
        {
            for (label j = 0; j < vector::nComponents; ++j)
            {
                sampledState[i * vector::nComponents + j] = sampledVector[i][j];
            }
        }
    }
    else
    {
        FatalErrorInFunction
            << "State field " << stateFieldName_ << " must be either"
            << "volScalarField or volVectorField"
            << exit(FatalError);
    }    
    
    // Normalize the field to range [-1, 1] using member data minVal_ and maxVal_
    if (stateMax_ != stateMin_)  // Avoid division by zero
    {
        forAll(sampledState, i)
        {
            sampledState[i] = 2 * ((sampledState[i] - stateMin_) / (stateMax_ - stateMin_)) - 1;
        }
    }
    else
    {
        WarningInFunction
            << "Normalization skipped due to identical min and max values for the state field."
            << endl;
    }

    return sampledState;
}


void Foam::agentJetFvPatchVectorField::loadModel()
{
    if ((modelType_ == "PyTorch" && !ptModel_) || (modelType_ == "TensorFlow" && !tfModel_))
    {
        fileName modelPath = db().time().globalPath() / policyDirName_;

        if (modelType_ == "PyTorch")
        {
            ptModel_.reset(new torch::jit::Module(torch::jit::load(modelPath / "policy.pt")));
            ptModel_->eval();
            Info << "PyTorch model loaded successfully from " << modelPath << endl;
        }
        else if (modelType_ == "TensorFlow")
        {
            tfModel_.reset(new cppflow::model(modelPath));
            Info << "TensorFlow model loaded successfully from " << modelPath << endl;
        }
        else
        {
            FatalErrorInFunction
                << "Invalid modelType '" << modelType_
                << "'. Supported values are 'TensorFlow' and 'PyTorch'."
                << abort(FatalError);
        }
    }
}


Foam::scalarField Foam::agentJetFvPatchVectorField::agentAction(const scalarField& state)
{
    scalarField rawAction;

    // if (modelType_ == "PyTorch")
    // {
    //     rawAction = agentActionPT(state);
    // }
    if (modelType_ == "TensorFlow")
    {
        rawAction = agentActionTF(state);
    }
    else
    {
        FatalErrorInFunction
            << "Invalid modelType '" << modelType_
            << "'. Supported values are 'TensorFlow' and 'PyTorch'."
            << abort(FatalError);
        return scalarField();
    }

    if (zeroMeanAction_ && nActions_ > 1)
    {
        rawAction = rawAction - sum(rawAction) / nActions_;
    }

    return rawAction;
}


Foam::scalar Foam::agentJetFvPatchVectorField::agentActionPT(const scalarField& state)
{
    std::vector<scalar> stateVec(state.begin(), state.end());

    // Convert state vector to Torch tensor
    torch::Tensor obs_tensor = torch::from_blob
    (
        stateVec.data(),
        {1, (long)state.size()},
        torch::kDouble
    ).clone(); // Clone to avoid issues with memory management

    // Feeding the inputs to the loaded model to create an output (action)
    torch::Tensor action_tensor = ptModel_->forward({obs_tensor, deterministic_}).toTensor();

    // Get the raw action value from the model
    scalar rawAction = action_tensor[0][0].item<scalar>();
    return rawAction;
}


Foam::scalarField Foam::agentJetFvPatchVectorField::agentActionTF(const scalarField& state)
{
    std::vector<float> stateVec(state.begin(), state.end());

    // Creating the input tensors of the policy model
    cppflow::tensor stateTensor(stateVec, {1, state.size()});
    std::vector<bool> det(1, deterministic_);
    cppflow::tensor detTensor(det, {});

    // Feeding the inputs to the loaded model to create an output (action)
    // The string arguements are found using saved_model_cli of Tensorflow
    auto action = tfModel_.ref()
    (
        {
            {"serving_default_args_0:0", stateTensor}, // Model input 0,
            {"serving_default_deterministic:0", detTensor} // Model input 1
        },
        {
            "StatefulPartitionedCall:0" // Model output
        }
    );

    // Get the raw action value from the model
    std::vector<float> rawActionVec = action[0].get_data<float>();
    scalarField rawAction(rawActionVec.size());
    forAll(rawAction, i)
    {
        rawAction[i] = rawActionVec[i];
    }

    return rawAction;
}


void Foam::agentJetFvPatchVectorField::initializeWriter()
{
    if (!writer_)
    {
        writer_.reset(new functionObjects::writeFile(db(), typeName, "ActionState", dict_));

        Ostream& os = writer_->file();
        writeFileHeader(os);
    }
}


void Foam::agentJetFvPatchVectorField::writeFileHeader(Ostream& os)
{
    writer_->writeHeader(os, "Trajectory actions and states");
    writer_->writeCommented(os, "Time");
    writer_->writeCommented(os, "Action(" + Foam::name(nActions_) + ")");
    writer_->writeCommented(os, "State (" + Foam::name(stateProbeLocations_.size()) + ")");
    os << endl;
}


void Foam::agentJetFvPatchVectorField::writeStateAction
(
    const scalarField& state,
    const scalarField actionNew
)
{
    initializeWriter();

    Ostream& os = writer_->file();
    writer_->writeCurrentTime(os);

    forAll(actionNew, i)
    {
        os  << tab << actionNew[i];
    }    
    forAll(state, i)
    {
        os  << tab << state[i];
    }
    os  << endl;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    // functionObjects::writeFile(db(), typeName, "ActionState"),
    dict_(),
    deterministic_(false),
    controlPeriod_(0),
    rampUpPeriod_(0),
    nActions_(1),
    zeroMeanAction_(false),
    faceActionMapping_(),
    actionNew_(),
    actionOld_(),
    jetDirection_(p.size()),
    curTimeIndex_(-1),
    stateFieldName_(),
    stateProbesNo_(),
    stateProbeLocations_(Zero),
    interpolationScheme_(),
    stateMin_(),
    stateMax_(),
    actionBound_(),
    policyDirName_(),
    modelType_(),
    tfModel_(),
    ptModel_()
{}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    // functionObjects::writeFile(ptf),
    dict_(ptf.dict_),
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    nActions_(ptf.nActions_),
    zeroMeanAction_(ptf.zeroMeanAction_),
    faceActionMapping_(ptf.faceActionMapping_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    jetDirection_(ptf.jetDirection_, mapper),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    stateMin_(ptf.stateMin_),
    stateMax_(ptf.stateMax_),
    actionBound_(ptf.actionBound_),
    policyDirName_(ptf.policyDirName_),
    modelType_(ptf.modelType_),
    tfModel_(ptf.tfModel_),
    ptModel_(ptf.ptModel_)
{}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    // functionObjects::writeFile(db(),typeName,"ActionState",dict),
    dict_(dict), // Store the dictionary
    deterministic_(dict.get<bool>("deterministic")),
    controlPeriod_(dict.get<scalar>("controlPeriod")),
    rampUpPeriod_(dict.get<scalar>("rampUpPeriod")),
    nActions_(dict.getOrDefault<label>("nActions", 1)),
    zeroMeanAction_(dict.getOrDefault<bool>("zeroMeanAction", false)),
    actionNew_("actionNew", dict, nActions_, IOobjectOption::LAZY_READ),
    actionOld_("actionOld", dict, nActions_, IOobjectOption::LAZY_READ),
    curTimeIndex_(-1),
    stateFieldName_(dict.get<word>("stateField")),
    stateProbesNo_(dict.get<label>("stateProbesNo")),
    interpolationScheme_(dict.get<word>("interpolationScheme")),
    stateMin_(dict.get<scalar>("stateMin")),
    stateMax_(dict.get<scalar>("stateMax")),
    actionBound_(dict.get<scalar>("actionBound")),
    policyDirName_(dict.get<fileName>("policyDir")),
    modelType_(dict.get<word>("modelType"))
{
    initializeFaceMapping();

    stateProbeLocations_ = vectorField
    (
        "stateProbeLocations",
        dict,
        stateProbesNo_
    );
     
    if (controlPeriod_ < rampUpPeriod_)
    {
        FatalErrorInFunction
            << "rampUpPeriod must be less that or equal to controlPeriod"
            << abort(FatalError);
    }
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        updateCoeffs();
    }
}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf
)
:
    fixedValueFvPatchField<vector>(ptf),
    // functionObjects::writeFile(ptf),
    dict_(ptf.dict_),
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    nActions_(ptf.nActions_),
    zeroMeanAction_(ptf.zeroMeanAction_),
    faceActionMapping_(ptf.faceActionMapping_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    jetDirection_(ptf.jetDirection_),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    stateMin_(ptf.stateMin_),
    stateMax_(ptf.stateMax_),
    actionBound_(ptf.actionBound_),
    policyDirName_(ptf.policyDirName_),
    modelType_(ptf.modelType_),
    tfModel_(ptf.tfModel_),
    ptModel_(ptf.ptModel_)
{}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF),
    // functionObjects::writeFile(ptf),
    dict_(ptf.dict_),
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    nActions_(ptf.nActions_),
    zeroMeanAction_(ptf.zeroMeanAction_),
    faceActionMapping_(ptf.faceActionMapping_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    jetDirection_(ptf.jetDirection_),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    stateMin_(ptf.stateMin_),
    stateMax_(ptf.stateMax_),
    actionBound_(ptf.actionBound_),
    policyDirName_(ptf.policyDirName_),
    modelType_(ptf.modelType_),
    tfModel_(ptf.tfModel_),
    ptModel_(ptf.ptModel_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::agentJetFvPatchVectorField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    fixedValueFvPatchField<vector>::autoMap(m);

    jetDirection_.autoMap(m);
}


void Foam::agentJetFvPatchVectorField::rmap
(
     const fvPatchVectorField& ptf,
     const labelList& addr
)
{
    fixedValueFvPatchField<vector>::rmap(ptf, addr);

    const auto& aj = dynamic_cast<const Foam::agentJetFvPatchVectorField&>(ptf);
    jetDirection_.rmap(aj.jetDirection_, addr);
}


void Foam::agentJetFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // Due to the inherent randomness of the agent's neural network, all the policy computations
    // are performed only once at each time step (e.g., the first outer corrector loop of PIMPLE)
    const label timeIndex = db().time().timeIndex();
    if (curTimeIndex_ != timeIndex)
    {
        const Time& time = db().time();
        scalar dt = time.deltaTValue();

        //TODO: What if period is not divisable by dt?
        const label nControlSteps = controlPeriod_ / dt;
        const label nRampSteps = rampUpPeriod_ / dt;

        scalarField currentAction(actionNew_);
        label currentControlStep = (timeIndex % nControlSteps + nControlSteps) % nControlSteps;

        if (currentControlStep == 1)
        {
            Info<< "Updating agent action with policy model" << endl;
            scalarField state = environmentState();

            // Agian, due to randomness of the model, the new action is computed only for the
            // master processor and broadcast to other processors
            if (Pstream::master())
            {
                loadModel();
                actionOld_ = actionNew_;
                actionNew_ = agentAction(state);
                writeStateAction(state, actionNew_);
                Info<< "New action = " 
                    << actionNew_
                    << ", Old action = "
                    << actionOld_ 
                    << endl;
            }
            // Broadcast the same action value on all processors when parallel processing
            Pstream::broadcast(actionNew_);
            Pstream::broadcast(actionOld_);
        }
        
        // Ramp up/down the new controller value
        scalar rampCoeff = 1;
        if ((currentControlStep <= nRampSteps) && (currentControlStep != 0))
        {
            rampCoeff = scalar(currentControlStep)/scalar(nRampSteps);
        }
        currentAction = rampCoeff*actionNew_ + (1 - rampCoeff)*actionOld_;

        tmp<vectorField> tvalues(new vectorField(patch().size(), vector::zero));
        vectorField& values = tvalues.ref();
        forAll(patch(), faceI)
        {
            label actionIndex = faceActionMapping_[faceI];
            values[faceI] = actionBound_ * currentAction[actionIndex] * jetDirection_[faceI];
        }

        vectorField::operator=(tvalues);

        curTimeIndex_ = db().time().timeIndex();
    }

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::agentJetFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry<bool>("deterministic", deterministic_);
    os.writeEntry("controlPeriod", controlPeriod_);
    os.writeEntry("rampUpPeriod", rampUpPeriod_);
    os.writeEntry("nActions", nActions_);
    os.writeEntry<bool>("zeroMeanAction", zeroMeanAction_);
    os.writeEntry<word>("policyDir", policyDirName_);
    os.writeEntry<word>("modelType", modelType_);
    actionNew_.writeEntry("actionNew", os);
    actionOld_.writeEntry("actionOld", os);
    faceActionMapping_.writeEntry("faceActionMapping", os);
    jetDirection_.writeEntry("jetDirections", os);
    os.writeEntry<word>("stateField", stateFieldName_);
    os.writeEntry("stateProbesNo", stateProbesNo_);
    stateProbeLocations_.writeEntry("stateProbeLocations", os);
    os.writeEntry<word>("interpolationScheme", interpolationScheme_);
    os.writeEntry("stateMin", stateMin_);
    os.writeEntry("stateMax", stateMax_);
    os.writeEntry("actionBound", actionBound_);
    writeEntry("value", os);
}
    

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        agentJetFvPatchVectorField
    );
}

// ************************************************************************* //
