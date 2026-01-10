use wgpu::{PollError, PollStatus, PollType, SubmissionIndex};

pub trait DeviceUtil {
    fn wait_for_submission(&self, submission_index: SubmissionIndex) -> Result<PollStatus, PollError>;
}

impl DeviceUtil for wgpu::Device {
    fn wait_for_submission(&self, submission_index: SubmissionIndex) -> Result<PollStatus, PollError> {
        self.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
    }
}
