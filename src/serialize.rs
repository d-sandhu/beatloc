//! JSON serialization boundary. The only place serde_json is used, so the
//! rest of the pipeline never depends on the output format.

use crate::BeatlocError;
use crate::timeline::Timeline;

pub fn to_json_string(timeline: &Timeline, pretty: bool) -> Result<String, BeatlocError> {
    let result = if pretty {
        serde_json::to_string_pretty(timeline)
    } else {
        serde_json::to_string(timeline)
    };
    result.map_err(|e| BeatlocError::Decode(format!("JSON serialization failed: {e}")))
}
